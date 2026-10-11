#!/usr/bin/env python3
"""GROW-style group-relative on-policy flow tuning of Chatterbox S3Gen LoRA.

This is not discrete-token GRPO: eight flow samples share one VC condition,
their two frozen Ears rewards are standardized separately, and signed group
advantages weight the flow loss with a frozen-policy velocity anchor.
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
import random
import sys
import tarfile
import time

import numpy as np
import pyarrow.parquet as pq
from safetensors.torch import load_file
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.nn import functional as F
from transformers import WhisperFeatureExtractor

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
from generate_listening import encode_mp3, load_model, ref_dict, set_adapter
from select_listening import TARGET_STATIC
from train import LoRALinear, StyledSpeakerProjection
from score_inputs import EARS, LayeredMultiTaskWhisper, to16

BASE = Path(os.environ.get('CHATTERBOX_RUN_ROOT',
    '/e/scratch/reformo/schuhmann1_moss/out/chatterbox_s3gen_300h_v1'))
OUT = BASE / 'grow_v1'
TRAIN_ROOT = Path(os.environ.get('CHATTERBOX_GROW_TRAIN_ROOT',
    '/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v1/training'))
GROUP = 8
NFE = 10
MAX_STEPS = 4000
WARMUP = 200
LR_PEAK = 5e-6
LR_FLOOR = 1e-6
BETA_ANCHOR = .025
REWARD_PQ = .55
REWARD_EMOTION = .45
CHECKPOINT_EVERY = 250
SEED = 20261009


class EarsReward:
    def __init__(self, device):
        self.device = device
        self.norm = json.loads((EARS / 'training_normalization.json').read_text())
        names = self.norm['score_names']
        self.pq_index = names.index('audiobox:PQ')
        self.names = names[:40]
        classes = json.loads((EARS / 'classes.json').read_text())['names']
        model = LayeredMultiTaskWhisper(EARS / 'medium', len(classes), initialize_pretrained=False)
        model.load_state_dict(load_file(EARS / 'medium/model.safetensors', device='cpu'), strict=True)
        self.model = model.to(device).eval().requires_grad_(False)
        self.extractor = WhisperFeatureExtractor.from_pretrained(EARS / 'medium', local_files_only=True)
        self.mean = np.asarray(self.norm['score_mean'], np.float32)
        self.std = np.asarray(self.norm['score_std'], np.float32)

    def score(self, waves24: np.ndarray) -> np.ndarray:
        assert waves24.ndim == 2 and waves24.shape[0] > 0
        waves16 = [to16(np.asarray(wave, np.float32), 24000) for wave in waves24]
        features = self.extractor(waves16, sampling_rate=16000, padding='longest', truncation=True,
                                  max_length=480000, return_attention_mask=True, return_tensors='np')
        mel = features['input_features'].astype(np.float32)
        mask = features['attention_mask'].astype(np.int64)
        if mel.shape[-1] % 2:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, 1)))
            mask = np.pad(mask, ((0, 0), (0, 1)))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = self.model(torch.from_numpy(mel).to(self.device),
                                torch.from_numpy(mask).to(self.device), predict_events=False)
        raw = output['scores'].float().cpu().numpy() * self.std + self.mean
        assert raw.shape == (waves24.shape[0], len(self.mean)) and np.isfinite(raw).all()
        return raw


def merge_and_reseed(model) -> float:
    """Fold the old q/k/v/o LoRA into base weights, then zero-init a new LoRA."""
    differences = []
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, LoRALinear):
                continue
            before = module.base.weight.float() + module.lora_up.weight.float() @ module.lora_down.weight.float()
            module.base.weight.add_((module.lora_up.weight @ module.lora_down.weight).to(module.base.weight))
            nn.init.kaiming_uniform_(module.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(module.lora_up.weight)
            after = module.base.weight.float() + module.lora_up.weight.float() @ module.lora_down.weight.float()
            differences.append(float((before - after).abs().max()))
    assert len(differences) == 224 and max(differences) < 1e-5, max(differences)
    return max(differences)


def conditioning_scores(source, target, score_order) -> np.ndarray:
    source_values = np.frombuffer(source['scores'], dtype='<f4').copy()
    target_values = np.frombuffer(target['scores'], dtype='<f4')
    assert source_values.shape == target_values.shape == (99,)
    for name in TARGET_STATIC:
        index = score_order.index(name)
        source_values[index] = target_values[index]
    return source_values


def flow_inputs(flow, token, token_len, camp, scores, prompt_mel):
    batch = token.shape[0]
    projection = flow.spk_embed_affine_layer
    assert isinstance(projection, StyledSpeakerProjection)
    projection.condition = scores
    projection.present = torch.ones_like(scores)
    embedding = projection(F.normalize(camp, dim=1))
    mask_tok = torch.ones((batch, token.shape[1], 1), device=token.device)
    embedded = flow.input_embedding(token.long()) * mask_tok
    encoded, lengths = flow.encoder(embedded, token_len)
    mu = flow.encoder_proj(encoded).transpose(1, 2).contiguous()
    assert mu.shape[1] == 80 and mu.shape[2] == 2 * token.shape[1]
    cond = torch.zeros_like(mu)
    cond[:, :, :prompt_mel.shape[2]] = prompt_mel
    return mu, embedding, cond


def grow_loss(flow, reference_flow, x1, token, token_len, camp, scores, prompt_len, advantage):
    batch = x1.shape[0]
    mu, embedding, cond = flow_inputs(flow, token, token_len, camp, scores, x1[:, :, :prompt_len])
    assert x1.shape == mu.shape and advantage.shape == (batch,)
    t = torch.rand((batch, 1, 1), device=x1.device)
    if flow.decoder.t_scheduler == 'cosine':
        t = 1 - torch.cos(t * math.pi * .5)
    noise = torch.randn_like(x1)
    sigma = float(flow.decoder.sigma_min)
    y = (1 - (1 - sigma) * t) * noise + t * x1
    velocity = x1 - (1 - sigma) * noise
    mask = torch.ones((batch, 1, x1.shape[2]), device=x1.device, dtype=x1.dtype)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = flow.decoder.estimator(y, mask, mu, t.flatten(), embedding, cond)
        with torch.no_grad():
            old_mu, old_embedding, old_cond = flow_inputs(
                reference_flow, token, token_len, camp, scores, x1[:, :, :prompt_len])
            old_pred = reference_flow.decoder.estimator(
                y, mask, old_mu, t.flatten(), old_embedding, old_cond)
    mse = (pred[:, :, prompt_len:].float() - velocity[:, :, prompt_len:]).square().mean((1, 2))
    anchor = (pred[:, :, prompt_len:].float() - old_pred[:, :, prompt_len:].float()).square().mean()
    signed = (mse * advantage.detach()).mean()
    loss = signed + BETA_ANCHOR * anchor
    return loss, float(signed.detach()), float(anchor.detach())


def rate(step: int) -> float:
    if step < WARMUP:
        return LR_PEAK * (step + 1) / WARMUP
    frac = min(1., (step - WARMUP) / max(1, MAX_STEPS - WARMUP))
    return LR_FLOOR + (LR_PEAK - LR_FLOOR) * .5 * (1 + math.cos(math.pi * frac))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    os.replace(temporary, path)


def checkpoint(model, optimizer, arm: str, step: int) -> None:
    directory = TRAIN_ROOT / arm / 'checkpoints'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'step-{step:07d}.pt'
    temporary = target.with_suffix('.pt.partial')
    state = {'schema': 'chatterbox-s3gen-grow-v1', 'arm': arm, 'step': step,
             'total_steps': MAX_STEPS, 'group_size': GROUP,
             'reward': {'audiobox_PQ': REWARD_PQ, 'negative_top3_emotion_mse': REWARD_EMOTION},
             'trainable': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
             'optimizer': optimizer.state_dict(), 'cpu_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state(),
             'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    torch.save(state, temporary)
    os.replace(temporary, target)
    atomic_json(directory / f'step-{step:07d}.COMPLETE.json',
                {'status': 'COMPLETE', 'step': step, 'arm': arm, 'file': str(target),
                 'bytes': target.stat().st_size})
    atomic_json(directory / 'LATEST.json', {'step': step, 'arm': arm, 'file': str(target)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'full'), required=True)
    parser.add_argument('--arm', choices=('continue', 'merged_new'), required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    rank = int(os.environ['SLURM_PROCID'])
    world = int(os.environ['SLURM_NTASKS'])
    assert (world == 2 and rank in (0, 1)) or (world == 1 and rank == 0)
    if world == 2:
        assert args.arm == ('continue' if rank == 0 else 'merged_new')
    device = torch.device('cuda:0' if torch.cuda.device_count() == 1 else 'cuda:' + os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    selection = json.loads((OUT / 'SELECTION.json').read_text())
    assert selection['status'] == 'SELECTED' and selection['pairs'] == MAX_STEPS
    assert json.loads((OUT / 'POOL_COMPLETE.json').read_text())['status'] == 'COMPLETE'
    for i in range(4):
        assert json.loads((OUT / f'input_ears_{i}.COMPLETE.json').read_text())['status'] == 'COMPLETE'
    pairs = pq.read_table(OUT / 'selection.parquet').to_pylist()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pairs)).tolist()
    pool = {r['uid']: r for r in pq.read_table(OUT / 'pool.parquet').to_pylist()}
    input_ears = {r['uid']: r for i in range(4) for r in pq.read_table(
        OUT / f'input_ears_{i}.parquet').to_pylist()}
    assert len(input_ears) == selection['unique_source_uids']
    score_order = json.loads((BASE / 'packed/train-0.COMPLETE.json').read_text())['score_order']
    scorer = EarsReward(device)
    assert scorer.names == ['emo_' + n for n in selection['emotion_axis_order']]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = load_model(device)
    reference_flow = copy.deepcopy(model.flow).eval().requires_grad_(False)
    init_merge_diff = merge_and_reseed(model) if args.arm == 'merged_new' else 0.0
    model.flow.decoder.training_cfg_rate = 0.0
    reference_flow.decoder.training_cfg_rate = 0.0
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and sum(p.numel() for p in trainable) > 10_000_000
    optimizer = torch.optim.AdamW(trainable, lr=LR_PEAK, weight_decay=.01)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert state['schema'] == 'chatterbox-s3gen-grow-v1' and state['arm'] == args.arm
        assert state['total_steps'] == MAX_STEPS and state['group_size'] == GROUP
        target = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert set(target) == set(state['trainable'])
        for name, param in target.items():
            param.data.copy_(state['trainable'][name].to(param.device))
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        start = int(state['step'])
    if args.mode == 'full' and not args.resume:
        # A new production run needs a passing smoke. A trusted, validated
        # optimizer-bearing resume checkpoint already proves the arm started.
        for arm in ('continue', 'merged_new'):
            proof = json.loads((OUT / 'smoke' / f'{arm}.json').read_text())
            assert proof['status'] == 'PASS' and proof['finite_steps'] == 2
    metrics_name = f'metrics_resume_from_{start:07d}.jsonl' if args.resume else 'metrics.jsonl'
    metrics_path = TRAIN_ROOT / args.arm / metrics_name
    if args.mode == 'full':
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
    end = min(start + 2, MAX_STEPS) if args.mode == 'smoke' else MAX_STEPS
    started = time.monotonic()
    previous_trainable = next(p for p in model.parameters() if p.requires_grad).detach().clone()
    for step in range(start, end):
        row = pairs[order[step]]
        source = pool[row['source_uid']]
        target = pool[row['target_uid']]
        scores = conditioning_scores(source, target, score_order)
        reference = ref_dict(target, device, acoustic_prompt=True)
        tokens_np = np.frombuffer(source['tokens'], dtype='<i2').astype(np.int64)
        speech_tokens = torch.from_numpy(tokens_np[None]).to(device).repeat(GROUP, 1)
        set_adapter(model, scores, 1.0, device)
        old_proj = reference_flow.spk_embed_affine_layer
        old_proj.condition = model.flow.spk_embed_affine_layer.condition
        old_proj.present = model.flow.spk_embed_affine_layer.present
        # Same per-step random stream in both arms; the effective policies
        # are checked to match at initialization in the two-step smoke.
        torch.manual_seed(SEED + step)
        torch.cuda.manual_seed_all(SEED + step)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            generated_mel = model.flow_inference(
                speech_tokens=speech_tokens, ref_dict=dict(reference), n_cfm_timesteps=NFE,
                finalize=True)
            waves, _ = model.hift_inference(generated_mel)
        assert generated_mel.shape[0] == waves.shape[0] == GROUP
        wave_np = waves.float().cpu().numpy().reshape(GROUP, -1)
        assert np.isfinite(wave_np).all() and np.abs(wave_np).max() < 4.0
        if args.mode == 'full' and (step == 0 or (step + 1) % 1000 == 0):
            # A few packed listening canaries allow human inspection without
            # creating thousands of per-sample files on the shared filesystem.
            payload, clipped = encode_mp3(wave_np[0])
            preview = TRAIN_ROOT / args.arm / 'preview.tar'
            with tarfile.open(preview, 'a' if preview.exists() else 'w') as archive:
                member = tarfile.TarInfo(f'step-{step+1:07d}-seed0.mp3')
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
                details = json.dumps({'step': step+1, 'arm': args.arm,
                                      'source_uid': row['source_uid'],
                                      'target_uid': row['target_uid'],
                                      'clipped_fraction': clipped}).encode()
                member = tarfile.TarInfo(f'step-{step+1:07d}-seed0.json')
                member.size = len(details)
                archive.addfile(member, io.BytesIO(details))
        reward_raw = scorer.score(wave_np)
        source_emo = np.frombuffer(input_ears[row['source_uid']]['ears_emotions_raw_f32'],
                                   dtype='<f4')
        assert source_emo.shape == (40,)
        top3 = np.argsort(source_emo)[-3:][::-1]
        pq_scores = reward_raw[:, scorer.pq_index]
        neg_mse = -np.mean((reward_raw[:, top3] - source_emo[top3]) ** 2, axis=1)
        def standardize(values):
            std = float(np.std(values))
            return (values - np.mean(values)) / max(std, 1e-3), std
        z_pq, sd_pq = standardize(pq_scores)
        z_emo, sd_emo = standardize(neg_mse)
        if sd_pq < 1e-4 and sd_emo < 1e-4:
            continue
        advantages = REWARD_PQ * z_pq + REWARD_EMOTION * z_emo
        advantages = np.clip(advantages, -2.5, 2.5)
        advantages -= advantages.mean()
        advantage = torch.from_numpy(advantages.astype(np.float32)).to(device)
        prompt_tokens = reference['prompt_token'].long().repeat(GROUP, 1)
        all_tokens = torch.cat([prompt_tokens, speech_tokens], dim=1)
        all_lengths = torch.full((GROUP,), all_tokens.shape[1], dtype=torch.long, device=device)
        camp = reference['embedding'].float().repeat(GROUP, 1)
        prompt_mel = reference['prompt_feat'].float().transpose(1, 2).repeat(GROUP, 1, 1)
        x1 = torch.cat([prompt_mel, generated_mel.detach().float()], dim=2)
        assert x1.shape[2] == 2 * all_tokens.shape[1]
        optimizer.param_groups[0]['lr'] = rate(step)
        optimizer.zero_grad(set_to_none=True)
        loss, signed, anchor = grow_loss(model.flow, reference_flow, x1, all_tokens,
                                        all_lengths, camp, model.flow.spk_embed_affine_layer.condition.repeat(GROUP, 1),
                                        prompt_mel.shape[2], advantage)
        assert torch.isfinite(loss), (step, signed, anchor)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
        row_metric = {'step': step+1, 'arm': args.arm, 'emotion_axis': row['emotion'],
                      'source_uid': row['source_uid'], 'target_uid': row['target_uid'],
                      'top3_ears_emotions': [scorer.names[int(i)] for i in top3],
                      'top3_source_raw': [float(source_emo[int(i)]) for i in top3],
                      'pq_mean': float(np.mean(pq_scores)), 'pq_std': sd_pq,
                      'top3_mse_mean': float(-np.mean(neg_mse)), 'top3_mse_std': sd_emo,
                      'signed_flow_loss': signed, 'anchor_loss': anchor,
                      'grad_norm': float(norm), 'lr': rate(step),
                      'elapsed_s': time.monotonic()-started}
        if args.mode == 'full':
            with metrics_path.open('a') as handle:
                handle.write(json.dumps(row_metric, allow_nan=False) + '\n')
        print(json.dumps(row_metric, allow_nan=False), flush=True)
        if args.mode == 'full' and ((step+1) % CHECKPOINT_EVERY == 0 or step+1 == end):
            checkpoint(model, optimizer, args.arm, step+1)
    delta = float((next(p for p in model.parameters() if p.requires_grad).detach() - previous_trainable).abs().sum())
    if args.mode == 'smoke':
        assert delta > 0, 'No LoRA weight update'
        result = {'status': 'PASS', 'arm': args.arm, 'finite_steps': 2,
                  'merge_max_weight_error': init_merge_diff, 'trainable_delta_l1': delta,
                  'elapsed_s': time.monotonic()-started}
        destination = OUT / 'smoke' / f'{args.arm}.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(destination, result)
        print(json.dumps(result), flush=True)
    elif end == MAX_STEPS:
        atomic_json(TRAIN_ROOT / args.arm / 'DONE.json',
                    {'status': 'COMPLETE', 'arm': args.arm, 'step': end,
                     'final_checkpoint': str(TRAIN_ROOT / args.arm / 'checkpoints' /
                                             f'step-{end:07d}.pt')})


if __name__ == '__main__':
    main()
