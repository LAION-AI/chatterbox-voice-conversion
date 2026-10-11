"""DDP S3Gen estimator q/k/v/out rank-128 LoRA + global-score conditioner."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
import pyarrow.parquet as pq
from safetensors.torch import load_file
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from chatterbox.models.s3gen import S3Gen

SC = Path(os.environ.get('CHATTERBOX_SCRATCH', '/e/scratch/reformo/schuhmann1_moss'))
ROOT = Path(os.environ.get('CHATTERBOX_RUN_ROOT', str(SC / 'out/chatterbox_s3gen_300h_v1')))
WEIGHTS = Path(os.environ.get('CHATTERBOX_S3GEN_WEIGHTS', str(
    SC / 'hfcache/hub/models--ResembleAI--chatterbox/snapshots/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18/s3gen.safetensors')))
HERE = Path(__file__).resolve().parent
RANK_LORA = 128
BATCH = 2
PEAK_LR = 1e-4


class LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.base.requires_grad_(False)
        self.lora_down = nn.Linear(base.in_features, RANK_LORA, bias=False)
        self.lora_up = nn.Linear(RANK_LORA, base.out_features, bias=False)
        self.scale = 1.0
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x, scale=1.0):
        return self.base(x) + (self.scale * scale) * self.lora_up(self.lora_down(x))


class StyledSpeakerProjection(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.style = nn.Sequential(nn.Linear(198, 256), nn.SiLU(), nn.Linear(256, 80))
        nn.init.zeros_(self.style[-1].weight)
        nn.init.zeros_(self.style[-1].bias)
        self.condition = None
        self.present = None
        self.scale = 1.0

    def forward(self, camp):
        assert self.condition is not None and self.present is not None
        features = torch.cat((self.condition * self.present, self.present), dim=-1)
        return self.base(camp) + self.scale * self.style(features)


class PilotModel(nn.Module):
    def __init__(self):
        super().__init__()
        original = S3Gen()
        missing, unexpected = original.load_state_dict(load_file(WEIGHTS), strict=False)
        assert not unexpected and set(missing) <= {'tokenizer.window'}, (missing, unexpected)
        original.requires_grad_(False)
        self.flow = original.flow
        estimator = self.flow.decoder.estimator
        names = []
        for name, module in estimator.named_modules():
            if name.endswith(('.attn1.to_q', '.attn1.to_k', '.attn1.to_v', '.attn1.to_out.0')):
                assert isinstance(module, nn.Linear), (name, type(module))
                names.append(name)
        assert len(names) >= 64 and len(names) % 4 == 0, names[:4]
        for name in names:
            parent_name, leaf = name.rsplit('.', 1)
            parent = estimator.get_submodule(parent_name)
            base = estimator.get_submodule(name)
            if isinstance(parent, (nn.ModuleList, nn.Sequential)):
                parent[int(leaf)] = LoRALinear(base)
            else:
                setattr(parent, leaf, LoRALinear(base))
        self.lora_modules = len(names)
        self.flow.spk_embed_affine_layer = StyledSpeakerProjection(self.flow.spk_embed_affine_layer)

    def forward(self, tokens, token_len, mel, mel_len, camp, scores, present):
        projection = self.flow.spk_embed_affine_layer
        projection.condition = scores
        projection.present = present
        return self.flow.compute_loss({
            'speech_token': tokens, 'speech_token_len': token_len,
            'speech_feat': mel, 'speech_feat_len': mel_len,
            'embedding': camp,
        }, mel.device)['loss']


def windows(rows):
    result = []
    for index, row in enumerate(rows):
        n = row['token_count']
        assert row['mel_frames'] == 2 * n
        start = 0
        while start < n:
            end = min(start + 200, n)
            if 0 < n - end < 50:
                end = n
            assert end - start >= 45 and end - start <= 250
            result.append((index, start, end))
            start = end
    return result


def batch_at(rows, order, step, rank, device):
    picks = [order[(step * BATCH + offset) % len(order)] for offset in range(BATCH)]
    samples = []
    for index, lo, hi in picks:
        row = rows[index]
        n = row['token_count']
        tok = np.frombuffer(row['tokens'], dtype='<i2')
        mel = np.frombuffer(row['mel'], dtype='<f2').reshape(80, 2 * n)
        camp = np.frombuffer(row['camp'], dtype='<f2')
        scores = np.frombuffer(row['scores'], dtype='<f4')
        assert len(tok) == n and len(camp) == 192 and len(scores) == 99
        samples.append((tok[lo:hi], mel[:, 2 * lo:2 * hi], camp, scores))
    max_n = max(len(item[0]) for item in samples)
    token = np.zeros((BATCH, max_n), dtype=np.int64)
    feat = np.zeros((BATCH, 80, 2 * max_n), dtype=np.float32)
    camp = np.zeros((BATCH, 192), dtype=np.float32)
    score = np.zeros((BATCH, 99), dtype=np.float32)
    lengths = []
    for i, (t, m, c, s) in enumerate(samples):
        token[i, :len(t)] = t
        feat[i, :, :m.shape[1]] = m
        camp[i] = c
        score[i] = s
        lengths.append(len(t))
    present = np.ones((BATCH, 99), dtype=np.float32)
    rng = random.Random(20261008 + 104729 * step + rank)
    for i in range(BATCH):
        for a, b in ((0, 40), (40, 97), (97, 98), (98, 99)):
            if rng.random() < .10:
                present[i, a:b] = 0
    return (
        torch.from_numpy(token).to(device),
        torch.tensor(lengths, device=device, dtype=torch.long),
        torch.from_numpy(feat).to(device),
        torch.tensor([2 * n for n in lengths], device=device, dtype=torch.long),
        torch.from_numpy(camp).to(device),
        torch.from_numpy(score).to(device),
        torch.from_numpy(present).to(device),
    )


def save_checkpoint(model, optimizer, step, total, out, rank):
    if rank != 0:
        return
    module = model.module
    trainable = {name: param.detach().cpu() for name, param in module.named_parameters() if param.requires_grad}
    state = {'step': step, 'total_steps': total, 'trainable': trainable,
             'optimizer': optimizer.state_dict(), 'cpu_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state(), 'rank_lora': RANK_LORA,
             'base_sha256': '2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e',
             'code_sha256': hashlib.sha256((HERE / 'train.py').read_bytes()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'step-{step:07d}.pt'
    temporary = out / f'step-{step:07d}.pt.partial'
    torch.save(state, temporary)
    os.replace(temporary, target)
    (out / f'step-{step:07d}.COMPLETE.json').write_text(json.dumps({
        'step': step, 'total_steps': total, 'file': str(target),
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'full'), required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    rank = int(os.environ['SLURM_PROCID'])
    world = int(os.environ['SLURM_NTASKS'])
    assert world == 4 and 0 <= rank < 4
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=os.environ['SLURM_LOCALID'])
    torch.set_num_threads(4)
    device = torch.device('cuda:0' if torch.cuda.device_count() == 1 else 'cuda:' + os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world)
    proofs = [json.loads((ROOT / 'packed' / f'train-{i}.COMPLETE.json').read_text()) for i in range(4)]
    assert all(p['status'] == 'COMPLETE' and p['rank'] == i and p['rows'] > 0 for i, p in enumerate(proofs))
    assert len({tuple(p['score_order']) for p in proofs}) == 1
    assert sum(p['hours'] for p in proofs) >= 297.0, 'Prepared hours below 99% gate'
    if args.mode == 'full':
        smoke = json.loads((ROOT / 'SMOKE_PASS.json').read_text())
        assert smoke['status'] == 'PASS' and smoke['rank_lora'] == RANK_LORA
    rows = pq.read_table(ROOT / 'packed' / f'train-{rank}.parquet').to_pylist()
    order = windows(rows)
    random.Random(20261008 + rank).shuffle(order)
    local_steps = math.ceil(len(order) / BATCH)
    count = torch.tensor(local_steps, device=device, dtype=torch.long)
    dist.all_reduce(count, op=dist.ReduceOp.MAX)
    total_steps = int(count.item())
    assert total_steps > 1000
    torch.manual_seed(20261008 + rank)
    model = PilotModel().to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(torch.isfinite(p).all() for p in trainable)
    model.flow.eval()
    wrapped = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(trainable, lr=PEAK_LR, weight_decay=.01)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert state['rank_lora'] == RANK_LORA and state['total_steps'] == total_steps
        for name, param in wrapped.module.named_parameters():
            if param.requires_grad:
                param.data.copy_(state['trainable'][name].to(param.device))
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        start = int(state['step'])
    if rank == 0:
        print(json.dumps({'mode': args.mode, 'rank_lora': RANK_LORA,
                          'lora_modules': wrapped.module.lora_modules,
                          'trainable_parameters': sum(p.numel() for p in trainable),
                          'total_steps': total_steps, 'rank_hours': [p['hours'] for p in proofs]}), flush=True)
    timings = []
    losses = []
    final = min(total_steps, start + 20) if args.mode == 'smoke' else total_steps
    for step in range(start, final):
        tick = time.monotonic()
        warmup = max(1, int(.05 * total_steps))
        if step < warmup:
            factor = (step + 1) / warmup
        else:
            factor = .1 + .9 * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
        optimizer.param_groups[0]['lr'] = PEAK_LR * factor
        tensors = batch_at(rows, order, step, rank, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = wrapped(*tensors)
        assert torch.isfinite(loss), (rank, step, float(loss))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        assert torch.isfinite(norm), (rank, step, float(norm))
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - tick
        timings.append(elapsed)
        losses.append(float(loss.detach()))
        if rank == 0 and ((step + 1) % 100 == 0 or step < 3):
            print(json.dumps({'step': step + 1, 'total': total_steps, 'loss': losses[-1],
                              'lr': optimizer.param_groups[0]['lr'], 'step_s': elapsed}), flush=True)
        if args.mode == 'full' and ((step + 1) % 2000 == 0 or step + 1 == total_steps):
            dist.barrier()
            save_checkpoint(wrapped, optimizer, step + 1, total_steps, ROOT / 'checkpoints', rank)
            dist.barrier()
    if args.mode == 'smoke':
        # Ignore cold-start steps; projected full run must fit one 11.5 h allocation.
        local_median = statistics.median(timings[5:])
        metric = torch.tensor(local_median, device=device)
        dist.all_reduce(metric, op=dist.ReduceOp.MAX)
        projected_hours = float(metric.item()) * total_steps / 3600 * 1.15
        status = 'PASS' if projected_hours <= 10.5 and all(math.isfinite(v) for v in losses) else 'FAIL'
        if rank == 0:
            summary = {'status': status, 'rank_lora': RANK_LORA, 'steps': len(losses),
                       'max_rank_median_step_s': float(metric.item()), 'projected_hours_including_15pct_overhead': projected_hours,
                       'total_steps': total_steps, 'final_loss': losses[-1]}
            (ROOT / 'SMOKE_PASS.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(summary), flush=True)
        assert status == 'PASS', f'Projected run too long: {projected_hours:.1f}h'
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
