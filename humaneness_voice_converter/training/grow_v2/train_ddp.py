#!/usr/bin/env python3
"""Four-GPU, 16-candidate continuation of the Chatterbox S3Gen VC RL pilot.

One condition is shared by four ranks; each samples four candidates. Rewards
are standardized across all 16 candidates, and trainable gradients are
averaged across ranks before identical AdamW updates. The two arms must run
as separate four-GPU jobs because they have different frozen anchors.
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

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'grow_v1'))
from generate_listening import encode_mp3, load_model, ref_dict, set_adapter
from train_grow import EarsReward, atomic_json, conditioning_scores, grow_loss, merge_and_reseed

BASE = Path(os.environ.get('CHATTERBOX_RUN_ROOT',
    '/e/scratch/reformo/schuhmann1_moss/out/chatterbox_s3gen_300h_v1'))
V1 = BASE / 'grow_v1'
V1_TRAIN = Path(os.environ.get('CHATTERBOX_GROW_TRAIN_ROOT',
    '/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v1/training'))
OUT = Path(os.environ.get('CHATTERBOX_GROW_V2_ROOT',
    '/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v2_ce'))
WORLD = 4
GROUP = 16
LOCAL_GROUP = GROUP // WORLD
STEPS = 24000
WARMUP = 1200
LR_PEAK = 2e-6
LR_FLOOR = 2e-7
ANCHOR = .025
NFE = 10
CHECKPOINT_EVERY = 1000
SEED = 20261010
# Both names exist in Ears. Do not silently switch between the EIV head and
# AudioBox's CE head: they are different measurements.
REWARD_SCORE = os.environ.get('CHATTERBOX_GROW_V2_REWARD_SCORE', 'audiobox:CE')
assert REWARD_SCORE in ('eiv_extra:score_content_enjoyment', 'audiobox:CE')


def initial_checkpoint(arm: str) -> Path:
    old = 'continue' if arm == 'continue' else 'merged_new'
    return V1_TRAIN / old / 'checkpoints/step-0004000.pt'


def rate(step: int) -> float:
    if step < WARMUP:
        return LR_PEAK * (step + 1) / WARMUP
    fraction = min(1., (step - WARMUP) / (STEPS - WARMUP))
    return LR_FLOOR + (LR_PEAK - LR_FLOOR) * .5 * (1 + math.cos(math.pi * fraction))


def load_start(model, arm: str) -> None:
    if arm == 'remerged_new':
        # Rebuild the v1 merged_new effective policy: merge the original
        # supervised LoRA, then load the first RL arm's fresh LoRA.
        merge_and_reseed(model)
    state = torch.load(initial_checkpoint(arm), map_location='cpu', weights_only=True)
    assert state['schema'] == 'chatterbox-s3gen-grow-v1' and state['step'] == 4000
    assert state['arm'] == ('continue' if arm == 'continue' else 'merged_new')
    target = {name: param for name, param in model.named_parameters() if param.requires_grad}
    assert set(target) == set(state['trainable'])
    for name, param in target.items():
        param.data.copy_(state['trainable'][name].to(param.device))


def sync_gradients(parameters: list[torch.nn.Parameter]) -> None:
    # Since each rank computes the mean loss over four candidates, averaging
    # the gradients yields the exact mean over the full 16-candidate group.
    grads = [p.grad for p in parameters]
    assert all(g is not None for g in grads), 'A trainable gradient is missing'
    flat = torch.cat([g.contiguous().view(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(WORLD)
    offset = 0
    for grad in grads:
        size = grad.numel()
        grad.copy_(flat[offset:offset + size].view_as(grad))
        offset += size
    assert offset == flat.numel()


def save_checkpoint(model, optimizer, arm: str, step: int) -> None:
    directory = OUT / arm / 'checkpoints'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'step-{step:07d}.pt'
    temporary = target.with_suffix('.pt.partial')
    state = {
        'schema': 'chatterbox-s3gen-grow-v2-ddp', 'arm': arm, 'step': step,
        'total_steps': STEPS, 'group_size': GROUP, 'world_size': WORLD,
        'start_checkpoint': str(initial_checkpoint(arm)),
        'reward': {'score': REWARD_SCORE, 'score_weight': .5,
                   'negative_top2_emotion_mse_weight': .5,
                   'group_standardized_separately': True},
        'lr': {'warmup_steps': WARMUP, 'peak': LR_PEAK, 'floor': LR_FLOOR,
               'schedule': 'cosine'},
        'trainable': {n: p.detach().cpu() for n, p in model.named_parameters()
                      if p.requires_grad},
        'optimizer': optimizer.state_dict(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    torch.save(state, temporary)
    os.replace(temporary, target)
    atomic_json(directory / f'step-{step:07d}.COMPLETE.json',
                {'status': 'COMPLETE', 'arm': arm, 'step': step,
                 'file': str(target), 'bytes': target.stat().st_size})
    atomic_json(directory / 'LATEST.json',
                {'step': step, 'arm': arm, 'file': str(target)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('continue', 'remerged_new'), required=True)
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
    # All ranks must initialize the *new* remerged LoRA identically before
    # gradient averaging; zero output at step zero does not imply matching
    # trainable down matrices.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = load_model(device)
    load_start(model, args.arm)
    # The anchor is the *respective completed v1 arm*, not the original SFT
    # model. Its output remains identical if we remerge the LoRA below.
    reference_flow = copy.deepcopy(model.flow).eval().requires_grad_(False)
    if args.arm == 'remerged_new':
        difference = merge_and_reseed(model)
        assert difference < 1e-5
    model.flow.decoder.training_cfg_rate = 0.0
    reference_flow.decoder.training_cfg_rate = 0.0
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) > 10_000_000
    optimizer = torch.optim.AdamW(trainable, lr=LR_PEAK, weight_decay=.01)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        assert state['schema'] == 'chatterbox-s3gen-grow-v2-ddp'
        assert state['arm'] == args.arm and state['group_size'] == GROUP
        assert state['world_size'] == WORLD and state['total_steps'] == STEPS
        assert state['reward']['score'] == REWARD_SCORE
        target = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert set(target) == set(state['trainable'])
        for name, param in target.items():
            param.data.copy_(state['trainable'][name].to(device))
        optimizer.load_state_dict(state['optimizer'])
        start = int(state['step'])
    # Fail before the first expensive rollout if any rank has different
    # trainable parameters (especially possible after a fresh LoRA remerge).
    fingerprint = torch.cat([p.detach().flatten()[:16] for p in trainable[:8]])
    copies = [torch.empty_like(fingerprint) for _ in range(WORLD)]
    dist.all_gather(copies, fingerprint)
    assert all(torch.equal(copies[0], part) for part in copies[1:])
    if args.mode == 'full' and start == 0:
        proof = json.loads((OUT / args.arm / 'SMOKE.json').read_text())
        assert proof['status'] == 'PASS' and proof['group_size'] == GROUP
        assert proof['reward_score'] == REWARD_SCORE
    end = min(start + 20, STEPS) if args.mode == 'smoke' else STEPS
    metric_path = OUT / args.arm / 'metrics.jsonl'
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
        source_emo = np.frombuffer(input_ears[row['source_uid']]['ears_emotions_raw_f32'],
                                   dtype='<f4')
        top2 = np.argsort(source_emo)[-2:][::-1]
        local_values = torch.tensor(np.stack([
            raw[:, score_index],
            -np.mean((raw[:, top2] - source_emo[top2]) ** 2, axis=1)
        ], axis=1), device=device, dtype=torch.float32)
        gathered = [torch.empty_like(local_values) for _ in range(WORLD)]
        dist.all_gather(gathered, local_values)
        values = torch.cat(gathered, dim=0)
        standard = (values - values.mean(0)) / values.std(0, unbiased=False).clamp_min(1e-3)
        advantage_full = (.5 * standard[:, 0] + .5 * standard[:, 1]).clamp(-2.5, 2.5)
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
        optimizer.param_groups[0]['lr'] = rate(step)
        optimizer.zero_grad(set_to_none=True)
        loss, signed, anchor = grow_loss(
            model.flow, reference_flow, x1, all_tokens, all_lengths, camp,
            model.flow.spk_embed_affine_layer.condition.repeat(LOCAL_GROUP, 1),
            prompt_mel.shape[2], advantage)
        assert torch.isfinite(loss), (step, signed, anchor)
        loss.backward()
        sync_gradients(trainable)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
        if rank == 0:
            metric = {'step': step + 1, 'arm': args.arm, 'reward_score': REWARD_SCORE,
                      'source_uid': row['source_uid'], 'target_uid': row['target_uid'],
                      'top2_emotions': [scorer.names[int(i)] for i in top2],
                      'quality_mean': float(values[:, 0].mean()),
                      'quality_std': float(values[:, 0].std(unbiased=False)),
                      'top2_mse_mean': float(-values[:, 1].mean()),
                      'top2_mse_std': float(values[:, 1].std(unbiased=False)),
                      'signed_flow_loss_rank0': signed,
                      'anchor_loss_rank0': anchor, 'grad_norm': float(norm),
                      'lr': rate(step), 'elapsed_s': time.monotonic() - started}
            if args.mode == 'full':
                with metric_path.open('a') as handle:
                    handle.write(json.dumps(metric, allow_nan=False) + '\n')
            if args.mode == 'smoke' or step < 3 or (step + 1) % 100 == 0:
                print(json.dumps(metric, allow_nan=False), flush=True)
            if args.mode == 'full' and (step == 0 or (step + 1) % 2000 == 0):
                audio, clipped = encode_mp3(wave_np[0])
                archive_path = OUT / args.arm / 'preview.tar'
                with tarfile.open(archive_path, 'a' if archive_path.exists() else 'w') as archive:
                    member = tarfile.TarInfo(f'step-{step+1:07d}.mp3')
                    member.size = len(audio)
                    archive.addfile(member, io.BytesIO(audio))
                    detail = json.dumps({'step': step+1, 'source_uid': row['source_uid'],
                                         'target_uid': row['target_uid'],
                                         'clipped_fraction': clipped}).encode()
                    member = tarfile.TarInfo(f'step-{step+1:07d}.json')
                    member.size = len(detail)
                    archive.addfile(member, io.BytesIO(detail))
            if args.mode == 'full' and ((step + 1) % CHECKPOINT_EVERY == 0 or step + 1 == end):
                save_checkpoint(model, optimizer, args.arm, step + 1)
        dist.barrier()
    delta = float((trainable[0].detach() - previous).abs().sum())
    if rank == 0:
        if args.mode == 'smoke':
            assert delta > 0
            result = {'status': 'PASS', 'arm': args.arm, 'group_size': GROUP,
                      'world_size': WORLD, 'groups_tested': end-start,
                      'reward_score': REWARD_SCORE, 'trainable_delta_l1': delta,
                      'elapsed_s': time.monotonic() - started}
            destination = OUT / args.arm / 'SMOKE.json'
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(destination, result)
            print(json.dumps(result), flush=True)
        else:
            atomic_json(OUT / args.arm / 'DONE.json',
                        {'status': 'COMPLETE', 'arm': args.arm, 'step': end,
                         'checkpoint': str(OUT / args.arm / 'checkpoints' /
                                           f'step-{end:07d}.pt')})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
