#!/usr/bin/env python3
"""16-candidate, four-GPU VC flow RL with CE/emotion/speaker reward.

Starts from the v1 `continue` step-4000 rank-128 LoRA. The decoder's frozen
CAMPPlus encoder measures target-voice cosine for the speaker reward.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.signal import resample_poly
import torch
import torch.distributed as dist
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'grow_v1'))
sys.path.insert(0, str(PROJECT / 'grow_v2'))
import train_ddp as shared
from generate_listening import encode_mp3, load_model, ref_dict, set_adapter
from train_grow import EarsReward, atomic_json, conditioning_scores, grow_loss

BASE = shared.BASE
V1 = shared.V1
OUT = Path(os.environ.get('CHATTERBOX_GROW_V3_ROOT',
    '/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v3_ce_emo_speaker'))
WORLD = shared.WORLD
GROUP = shared.GROUP
LOCAL_GROUP = shared.LOCAL_GROUP
STEPS = shared.STEPS
NFE = shared.NFE
SEED = shared.SEED
CHECKPOINT_EVERY = shared.CHECKPOINT_EVERY
REWARD_SCORE = 'audiobox:CE'  # Humaneness Ears Medium output axis
WEIGHTS = (.40, .40, .20)  # CE, negative top-two emotion MSE, target CAMPPlus


def speaker_cosines(model, wave24: np.ndarray, target: dict, device: torch.device) -> np.ndarray:
    assert wave24.ndim == 2 and wave24.shape[0] == LOCAL_GROUP
    wave16 = resample_poly(wave24, 2, 3, axis=1).astype(np.float32)
    reference = np.frombuffer(target['camp'], dtype='<f2').astype(np.float32)
    assert reference.shape == (192,)
    reference /= max(float(np.linalg.norm(reference)), 1e-8)
    with torch.inference_mode():
        embeddings = model.speaker_encoder.inference(
            torch.from_numpy(wave16).to(device)).float()
        embeddings = F.normalize(embeddings, dim=1).cpu().numpy()
    assert embeddings.shape == (LOCAL_GROUP, 192)
    scores = embeddings @ reference
    assert np.isfinite(scores).all()
    return scores


def checkpoint(model, optimizer, step: int) -> None:
    directory = OUT / 'speaker_mix/checkpoints'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'step-{step:07d}.pt'
    temporary = target.with_suffix('.pt.partial')
    state = {
        'schema': 'chatterbox-s3gen-grow-v3-ce-emo-speaker',
        'arm': 'speaker_mix', 'step': step, 'total_steps': STEPS,
        'group_size': GROUP, 'world_size': WORLD,
        'start_checkpoint': str(shared.initial_checkpoint('continue')),
        'reward': {'score': REWARD_SCORE, 'content_enjoyment_weight': WEIGHTS[0],
                   'negative_top2_emotion_mse_weight': WEIGHTS[1],
                   'target_campplus_cosine_weight': WEIGHTS[2],
                   'group_standardized_separately': True},
        'lr': {'warmup_steps': shared.WARMUP, 'peak': shared.LR_PEAK,
               'floor': shared.LR_FLOOR, 'schedule': 'cosine'},
        'trainable': {n: p.detach().cpu() for n, p in model.named_parameters()
                      if p.requires_grad},
        'optimizer': optimizer.state_dict(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    torch.save(state, temporary)
    os.replace(temporary, target)
    atomic_json(directory / f'step-{step:07d}.COMPLETE.json',
                {'status': 'COMPLETE', 'arm': 'speaker_mix', 'step': step,
                 'file': str(target), 'bytes': target.stat().st_size})
    atomic_json(directory / 'LATEST.json',
                {'arm': 'speaker_mix', 'step': step, 'file': str(target)})


def main() -> None:
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
    model = load_model(device)
    shared.load_start(model, 'continue')
    reference_flow = copy.deepcopy(model.flow).eval().requires_grad_(False)
    model.flow.decoder.training_cfg_rate = 0.0
    reference_flow.decoder.training_cfg_rate = 0.0
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) > 10_000_000
    optimizer = torch.optim.AdamW(trainable, lr=shared.LR_PEAK, weight_decay=.01)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        assert state['schema'] == 'chatterbox-s3gen-grow-v3-ce-emo-speaker'
        assert state['arm'] == 'speaker_mix' and state['total_steps'] == STEPS
        assert state['group_size'] == GROUP and state['world_size'] == WORLD
        target_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert set(target_params) == set(state['trainable'])
        for name, param in target_params.items():
            param.data.copy_(state['trainable'][name].to(device))
        optimizer.load_state_dict(state['optimizer'])
        start = int(state['step'])
    fingerprints = [torch.empty(128, device=device) for _ in range(WORLD)]
    fingerprint = torch.cat([p.detach().flatten()[:16] for p in trainable[:8]])
    dist.all_gather(fingerprints, fingerprint)
    assert all(torch.equal(fingerprints[0], part) for part in fingerprints[1:])
    if args.mode == 'full' and start == 0:
        proof = json.loads((OUT / 'speaker_mix/SMOKE.json').read_text())
        assert proof['status'] == 'PASS' and proof['group_size'] == GROUP
    end = min(start + 20, STEPS) if args.mode == 'smoke' else STEPS
    metric_path = OUT / 'speaker_mix/metrics.jsonl'
    if rank == 0 and args.mode == 'full':
        metric_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    previous = trainable[0].detach().clone()
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
        old_projection = reference_flow.spk_embed_affine_layer
        old_projection.condition = model.flow.spk_embed_affine_layer.condition
        old_projection.present = model.flow.spk_embed_affine_layer.present
        torch.manual_seed(SEED + step * WORLD + rank)
        torch.cuda.manual_seed_all(SEED + step * WORLD + rank)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated_mel = model.flow_inference(
                speech_tokens=speech_tokens, ref_dict=dict(reference),
                n_cfm_timesteps=NFE, finalize=True)
            waves, _ = model.hift_inference(generated_mel)
        assert generated_mel.shape[0] == waves.shape[0] == LOCAL_GROUP
        wave_np = waves.float().cpu().numpy().reshape(LOCAL_GROUP, -1)
        assert np.isfinite(wave_np).all() and np.abs(wave_np).max() < 4.0
        raw = scorer.score(wave_np)
        speaker = speaker_cosines(model, wave_np, target, device)
        source_emo = np.frombuffer(input_ears[row['source_uid']]['ears_emotions_raw_f32'],
                                   dtype='<f4')
        top2 = np.argsort(source_emo)[-2:][::-1]
        local_values = torch.tensor(np.stack([
            raw[:, score_index],
            -np.mean((raw[:, top2] - source_emo[top2]) ** 2, axis=1),
            speaker,
        ], axis=1), device=device, dtype=torch.float32)
        gathered = [torch.empty_like(local_values) for _ in range(WORLD)]
        dist.all_gather(gathered, local_values)
        values = torch.cat(gathered, dim=0)
        standard = (values - values.mean(0)) / values.std(0, unbiased=False).clamp_min(1e-3)
        advantage_full = (WEIGHTS[0] * standard[:, 0] +
                          WEIGHTS[1] * standard[:, 1] +
                          WEIGHTS[2] * standard[:, 2]).clamp(-2.5, 2.5)
        advantage_full -= advantage_full.mean()
        advantage = advantage_full[rank * LOCAL_GROUP:(rank + 1) * LOCAL_GROUP]
        prompt_tokens = reference['prompt_token'].long().repeat(LOCAL_GROUP, 1)
        all_tokens = torch.cat([prompt_tokens, speech_tokens], dim=1)
        all_lengths = torch.full((LOCAL_GROUP,), all_tokens.shape[1],
                                 dtype=torch.long, device=device)
        camp = reference['embedding'].float().repeat(LOCAL_GROUP, 1)
        prompt_mel = reference['prompt_feat'].float().transpose(1, 2).repeat(LOCAL_GROUP, 1, 1)
        x1 = torch.cat([prompt_mel, generated_mel.detach().float()], dim=2)
        assert x1.shape[2] == 2 * all_tokens.shape[1]
        optimizer.param_groups[0]['lr'] = shared.rate(step)
        optimizer.zero_grad(set_to_none=True)
        loss, signed, anchor = grow_loss(
            model.flow, reference_flow, x1, all_tokens, all_lengths, camp,
            model.flow.spk_embed_affine_layer.condition.repeat(LOCAL_GROUP, 1),
            prompt_mel.shape[2], advantage)
        assert torch.isfinite(loss), (step, signed, anchor)
        loss.backward()
        shared.sync_gradients(trainable)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
        if rank == 0:
            metric = {'step': step + 1, 'arm': 'speaker_mix',
                      'reward_score': REWARD_SCORE, 'reward_weights': WEIGHTS,
                      'source_uid': row['source_uid'], 'target_uid': row['target_uid'],
                      'top2_emotions': [scorer.names[int(i)] for i in top2],
                      'content_enjoyment_mean': float(values[:, 0].mean()),
                      'content_enjoyment_std': float(values[:, 0].std(unbiased=False)),
                      'top2_mse_mean': float(-values[:, 1].mean()),
                      'speaker_cosine_mean': float(values[:, 2].mean()),
                      'speaker_cosine_std': float(values[:, 2].std(unbiased=False)),
                      'signed_flow_loss_rank0': signed,
                      'anchor_loss_rank0': anchor, 'grad_norm': float(norm),
                      'lr': shared.rate(step), 'elapsed_s': time.monotonic() - started}
            if args.mode == 'full':
                with metric_path.open('a') as handle:
                    handle.write(json.dumps(metric, allow_nan=False) + '\n')
            if args.mode == 'smoke' or step < 3 or (step + 1) % 100 == 0:
                print(json.dumps(metric, allow_nan=False), flush=True)
            if args.mode == 'full' and (step == 0 or (step + 1) % 2000 == 0):
                audio, clipped = encode_mp3(wave_np[0])
                archive_path = OUT / 'speaker_mix/preview.tar'
                with tarfile.open(archive_path, 'a' if archive_path.exists() else 'w') as archive:
                    item = tarfile.TarInfo(f'step-{step+1:07d}.mp3')
                    item.size = len(audio)
                    archive.addfile(item, io.BytesIO(audio))
                    detail = json.dumps({'step': step+1, 'source_uid': row['source_uid'],
                                         'target_uid': row['target_uid'],
                                         'clipped_fraction': clipped}).encode()
                    item = tarfile.TarInfo(f'step-{step+1:07d}.json')
                    item.size = len(detail)
                    archive.addfile(item, io.BytesIO(detail))
            if args.mode == 'full' and ((step + 1) % CHECKPOINT_EVERY == 0 or step + 1 == end):
                checkpoint(model, optimizer, step + 1)
        dist.barrier()
    delta = float((trainable[0].detach() - previous).abs().sum())
    if rank == 0:
        if args.mode == 'smoke':
            assert delta > 0
            proof = {'status': 'PASS', 'arm': 'speaker_mix', 'group_size': GROUP,
                     'world_size': WORLD, 'groups_tested': end-start,
                     'reward': {'score': REWARD_SCORE, 'weights': WEIGHTS},
                     'trainable_delta_l1': delta,
                     'elapsed_s': time.monotonic() - started}
            destination = OUT / 'speaker_mix/SMOKE.json'
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(destination, proof)
            print(json.dumps(proof), flush=True)
        else:
            atomic_json(OUT / 'speaker_mix/DONE.json',
                        {'status': 'COMPLETE', 'arm': 'speaker_mix', 'step': end,
                         'checkpoint': str(OUT / 'speaker_mix/checkpoints' /
                                           f'step-{end:07d}.pt')})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
