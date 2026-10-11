#!/usr/bin/env python3
"""Full-flow continuation of merged Speaker-Mix: 40/30/30 GROW reward.

The upstream tokenizer, CAMPPlus speaker encoder and HiFT vocoder stay frozen.
All flow-generator weights, including the DiT and score-conditioning projection,
are trained. This is full fine-tuning of the VC generator, not another LoRA.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'grow_v1'))
sys.path.insert(0, str(PROJECT / 'grow_v2'))
sys.path.insert(0, str(PROJECT / 'grow_v3'))
import train_ddp as shared
from generate_listening import encode_mp3, load_model, ref_dict, set_adapter
from train import LoRALinear, StyledSpeakerProjection
from train_grow import EarsReward, atomic_json, conditioning_scores, grow_loss
from train_speaker import speaker_cosines

BASE = shared.BASE
V1 = shared.V1
V3 = Path(os.environ.get('CHATTERBOX_GROW_V3_ROOT',
    '/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v3_ce_emo_speaker'))
OUT = Path(os.environ.get('CHATTERBOX_GROW_V4_ROOT',
    '/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v4_full_403030'))
SOURCE = V3 / 'speaker_mix/checkpoints/step-0024000.pt'
WORLD, GROUP = 4, 16
LOCAL_GROUP = GROUP // WORLD
STEPS = 40000
PER_JOB = 8000
WARMUP = 2000
LR_PEAK = 3e-7
LR_FLOOR = 3e-8
CHECKPOINT_EVERY = 1000
SEED = 20261010
NFE = 10
WEIGHTS = (.40, .30, .30)
REWARD_SCORE = 'audiobox:CE'


def rate(step: int) -> float:
    if step < WARMUP:
        return LR_PEAK * (step + 1) / WARMUP
    fraction = min(1., (step - WARMUP) / (STEPS - WARMUP))
    return LR_FLOOR + (LR_PEAK - LR_FLOOR) * .5 * (1 + math.cos(math.pi * fraction))


def normalize_rollout_caches(flow) -> int:
    """Turn cached positional encodings from S3Gen's inference decorator into normal tensors.

    `S3Gen.flow_inference` is itself decorated with inference_mode, even when
    called inside no_grad. Its encoder caches a relative positional embedding
    in `module.pe`; full-flow backprop through linear_pos must save that input.
    """
    replaced = 0
    for module in flow.modules():
        pe = getattr(module, 'pe', None)
        if isinstance(pe, torch.Tensor) and pe.is_inference():
            module.pe = pe.clone()
            replaced += 1
    return replaced


def load_merged_model(device: torch.device):
    """Reconstruct the exact v3 policy, fold its 224 LoRAs, then unfreeze flow."""
    model = load_model(device)
    shared.load_start(model, 'continue')
    state = torch.load(SOURCE, map_location='cpu', weights_only=True)
    assert state['schema'] == 'chatterbox-s3gen-grow-v3-ce-emo-speaker'
    assert state['arm'] == 'speaker_mix' and state['step'] == 24000
    assert state['reward']['target_campplus_cosine_weight'] == .20
    target = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert set(target) == set(state['trainable'])
    with torch.no_grad():
        for name, p in target.items():
            p.copy_(state['trainable'][name].to(device))
    merged = []
    with torch.no_grad():
        for name, module in list(model.named_modules()):
            if not isinstance(module, LoRALinear):
                continue
            assert module.scale == 1.0
            parent_name, leaf = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            base = module.base
            x = torch.randn(2, base.in_features, device=device)
            before = module(x).float()
            base.weight.add_((module.lora_up.weight @ module.lora_down.weight).to(base.weight))
            after = base(x).float()
            error = float((before - after).abs().max())
            assert error < 1e-4, (name, error)
            if isinstance(parent, (nn.Sequential, nn.ModuleList)):
                parent[int(leaf)] = base
            else:
                setattr(parent, leaf, base)
            merged.append(error)
    assert len(merged) == 224 and not any(isinstance(m, LoRALinear) for m in model.modules())
    projection = model.flow.spk_embed_affine_layer
    assert isinstance(projection, StyledSpeakerProjection)
    model.requires_grad_(False)
    model.flow.requires_grad_(True)
    model.flow.decoder.training_cfg_rate = 0.0
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert any('.decoder.estimator.' in n for n in trainable)
    assert any('.spk_embed_affine_layer.style.' in n for n in trainable)
    assert any('.encoder.' in n for n in trainable)
    assert all(n.startswith('flow.') for n in trainable)
    return model, max(merged), sum(p.numel() for p in trainable.values())


def save_checkpoint(model, optimizer, step: int, trainable_names: list[str]):
    directory = OUT / 'checkpoints'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'step-{step:07d}.pt'
    partial = path.with_suffix('.pt.partial')
    state = {
        'schema': 'chatterbox-s3gen-grow-v4-full-403030',
        'step': step, 'total_steps': STEPS, 'group_size': GROUP,
        'world_size': WORLD, 'source': str(SOURCE),
        'reward': {'content_enjoyment': WEIGHTS[0], 'negative_top2_emotion_mse': WEIGHTS[1],
                   'target_campplus_cosine': WEIGHTS[2], 'group_standardized_separately': True},
        'lr': {'warmup_steps': WARMUP, 'peak': LR_PEAK, 'floor': LR_FLOOR,
               'schedule': 'single_cosine_over_40000'},
        'trainable': {n: p.detach().cpu() for n, p in model.named_parameters()
                      if n in trainable_names},
        'optimizer': optimizer.state_dict(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    assert set(state['trainable']) == set(trainable_names)
    torch.save(state, partial)
    os.replace(partial, path)
    atomic_json(directory / f'step-{step:07d}.COMPLETE.json',
                {'status': 'COMPLETE', 'step': step, 'file': str(path),
                 'bytes': path.stat().st_size})
    atomic_json(directory / 'LATEST.json', {'step': step, 'file': str(path)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'full'), required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    rank = int(os.environ['SLURM_PROCID'])
    world = int(os.environ['SLURM_NTASKS'])
    assert world == WORLD and 0 <= rank < WORLD
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world),
                      LOCAL_RANK=os.environ['SLURM_LOCALID'])
    torch.set_num_threads(4)
    device = torch.device('cuda:0' if torch.cuda.device_count() == 1
                          else 'cuda:' + os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', rank=rank, world_size=world)
    selection = json.loads((V1 / 'SELECTION.json').read_text())
    assert selection['status'] == 'SELECTED' and selection['pairs'] == 4000
    assert json.loads((V1 / 'POOL_COMPLETE.json').read_text())['status'] == 'COMPLETE'
    pairs = pq.read_table(V1 / 'selection.parquet').to_pylist()
    assert len(pairs) == 4000
    pool = {row['uid']: row for row in pq.read_table(V1 / 'pool.parquet').to_pylist()}
    input_ears = {row['uid']: row for i in range(4) for row in
                  pq.read_table(V1 / f'input_ears_{i}.parquet').to_pylist()}
    assert len(input_ears) == selection['unique_source_uids']
    score_order = json.loads((BASE / 'packed/train-0.COMPLETE.json').read_text())['score_order']
    scorer = EarsReward(device)
    score_index = scorer.norm['score_names'].index(REWARD_SCORE)
    assert scorer.names == ['emo_' + n for n in selection['emotion_axis_order']]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model, merge_error, n_params = load_merged_model(device)
    reference_flow = copy.deepcopy(model.flow).eval().requires_grad_(False)
    reference_flow.decoder.training_cfg_rate = 0.0
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    names = list(trainable)
    parameters = list(trainable.values())
    optimizer = torch.optim.AdamW(parameters, lr=LR_PEAK, weight_decay=.01)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        assert state['schema'] == 'chatterbox-s3gen-grow-v4-full-403030'
        assert state['group_size'] == GROUP and state['world_size'] == WORLD
        assert state['total_steps'] == STEPS and set(state['trainable']) == set(names)
        assert state['reward']['target_campplus_cosine'] == WEIGHTS[2]
        with torch.no_grad():
            for name, p in trainable.items():
                p.copy_(state['trainable'][name].to(device))
        optimizer.load_state_dict(state['optimizer'])
        start = int(state['step'])
    fingerprint = torch.cat([p.detach().flatten()[:16] for p in parameters[:8]])
    copies = [torch.empty_like(fingerprint) for _ in range(WORLD)]
    dist.all_gather(copies, fingerprint)
    assert all(torch.equal(copies[0], c) for c in copies[1:])
    if args.mode == 'full' and start == 0:
        smoke = json.loads((OUT / 'SMOKE.json').read_text())
        assert smoke['status'] == 'PASS' and smoke['group_size'] == GROUP
    end = min(start + (16 if args.mode == 'smoke' else PER_JOB), STEPS)
    if rank == 0:
        print(json.dumps({'event': 'START', 'mode': args.mode, 'start': start,
                          'end': end, 'flow_trainable_parameters': n_params,
                          'merge_max_abs_error': merge_error,
                          'peak_lr': LR_PEAK, 'weights': WEIGHTS}), flush=True)
    metric_path = OUT / 'metrics.jsonl'
    if rank == 0 and args.mode == 'full':
        metric_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    first = parameters[0].detach().clone()
    permutations = [np.random.default_rng(SEED + cycle).permutation(4000)
                    for cycle in range(STEPS // 4000)]
    for step in range(start, end):
        row = pairs[int(permutations[step // 4000][step % 4000])]
        source, target = pool[row['source_uid']], pool[row['target_uid']]
        scores = conditioning_scores(source, target, score_order)
        reference = ref_dict(target, device, acoustic_prompt=True)
        tokens = np.frombuffer(source['tokens'], dtype='<i2').astype(np.int64)
        speech_tokens = torch.from_numpy(tokens[None]).to(device).repeat(LOCAL_GROUP, 1)
        set_adapter(model, scores, 1.0, device)
        reference_flow.spk_embed_affine_layer.condition = model.flow.spk_embed_affine_layer.condition
        reference_flow.spk_embed_affine_layer.present = model.flow.spk_embed_affine_layer.present
        torch.manual_seed(SEED + step * WORLD + rank)
        torch.cuda.manual_seed_all(SEED + step * WORLD + rank)
        # S3Gen.flow_inference has its own inference_mode decorator, so its
        # output and cached positional encodings need normalization below.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated_mel = model.flow_inference(
                speech_tokens=speech_tokens, ref_dict=dict(reference),
                n_cfm_timesteps=NFE, finalize=True)
            waves, _ = model.hift_inference(generated_mel)
        generated_mel = generated_mel.clone().detach()
        normalized_caches = normalize_rollout_caches(model.flow)
        if rank == 0 and step == start:
            print(json.dumps({'event': 'ROLLOUT_CACHE_NORMALIZED',
                              'count': normalized_caches}), flush=True)
        assert generated_mel.shape[0] == waves.shape[0] == LOCAL_GROUP
        wave_np = waves.float().cpu().numpy().reshape(LOCAL_GROUP, -1)
        assert np.isfinite(wave_np).all() and np.abs(wave_np).max() < 4.0
        raw = scorer.score(wave_np)
        speaker = speaker_cosines(model, wave_np, target, device)
        source_emo = np.frombuffer(input_ears[row['source_uid']]['ears_emotions_raw_f32'],
                                   dtype='<f4')
        top2 = np.argsort(source_emo)[-2:][::-1]
        values_local = torch.tensor(np.stack([
            raw[:, score_index],
            -np.mean((raw[:, top2] - source_emo[top2]) ** 2, axis=1),
            speaker,
        ], axis=1), device=device, dtype=torch.float32)
        gathered = [torch.empty_like(values_local) for _ in range(WORLD)]
        dist.all_gather(gathered, values_local)
        values = torch.cat(gathered, dim=0)
        standard = (values - values.mean(0)) / values.std(0, unbiased=False).clamp_min(1e-3)
        advantages = (WEIGHTS[0] * standard[:, 0] + WEIGHTS[1] * standard[:, 1] +
                      WEIGHTS[2] * standard[:, 2]).clamp(-2.5, 2.5)
        advantages -= advantages.mean()
        advantage = advantages[rank * LOCAL_GROUP:(rank + 1) * LOCAL_GROUP]
        prompt_tokens = reference['prompt_token'].long().repeat(LOCAL_GROUP, 1)
        all_tokens = torch.cat([prompt_tokens, speech_tokens], dim=1)
        all_lengths = torch.full((LOCAL_GROUP,), all_tokens.shape[1], dtype=torch.long, device=device)
        camp = reference['embedding'].float().repeat(LOCAL_GROUP, 1)
        prompt_mel = reference['prompt_feat'].float().transpose(1, 2).repeat(LOCAL_GROUP, 1, 1)
        x1 = torch.cat([prompt_mel, generated_mel.detach().float()], dim=2)
        assert x1.shape[2] == 2 * all_tokens.shape[1]
        optimizer.param_groups[0]['lr'] = rate(step)
        optimizer.zero_grad(set_to_none=True)
        loss, signed, anchor = grow_loss(
            model.flow, reference_flow, x1, all_tokens, all_lengths, camp,
            model.flow.spk_embed_affine_layer.condition.repeat(LOCAL_GROUP, 1),
            prompt_mel.shape[2], advantage)
        assert torch.isfinite(loss), (step, signed, anchor)
        loss.backward()
        shared.sync_gradients(parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
        if rank == 0:
            metric = {'step': step + 1, 'content_enjoyment_mean': float(values[:, 0].mean()),
                      'top2_mse_mean': float(-values[:, 1].mean()),
                      'speaker_cosine_mean': float(values[:, 2].mean()),
                      'signed_flow_loss_rank0': signed, 'anchor_loss_rank0': anchor,
                      'grad_norm': float(norm), 'lr': rate(step),
                      'elapsed_s_this_job': time.monotonic() - started}
            if args.mode == 'full':
                with metric_path.open('a') as handle:
                    handle.write(json.dumps(metric, allow_nan=False) + '\n')
            if args.mode == 'smoke' or step < start + 3 or (step + 1) % 100 == 0:
                print(json.dumps(metric, allow_nan=False), flush=True)
            if args.mode == 'full' and (step == start or (step + 1) % 2000 == 0):
                audio, clipped = encode_mp3(wave_np[0])
                archive_path = OUT / 'preview.tar'
                with tarfile.open(archive_path, 'a' if archive_path.exists() else 'w') as archive:
                    for suffix, payload in [('mp3', audio), ('json', json.dumps(
                        {'step': step + 1, 'source_uid': row['source_uid'],
                         'target_uid': row['target_uid'], 'clipped_fraction': clipped}).encode())]:
                        item = tarfile.TarInfo(f'step-{step+1:07d}.{suffix}')
                        item.size = len(payload)
                        archive.addfile(item, io.BytesIO(payload))
            if args.mode == 'full' and ((step + 1) % CHECKPOINT_EVERY == 0 or step + 1 == end):
                save_checkpoint(model, optimizer, step + 1, names)
        dist.barrier()
    if rank == 0:
        if args.mode == 'smoke':
            delta = float((parameters[0].detach() - first).abs().sum())
            assert delta > 0
            atomic_json(OUT / 'SMOKE.json',
                        {'status': 'PASS', 'groups_tested': end - start,
                         'group_size': GROUP, 'world_size': WORLD,
                         'flow_trainable_parameters': n_params,
                         'merge_max_abs_error': merge_error, 'delta_l1': delta,
                         'elapsed_s': time.monotonic() - started})
        elif end == STEPS:
            atomic_json(OUT / 'DONE.json',
                        {'status': 'COMPLETE', 'step': STEPS,
                         'checkpoint': str(OUT / 'checkpoints' / f'step-{STEPS:07d}.pt')})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
