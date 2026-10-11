#!/usr/bin/env python3
"""Measure real GPU batching and numeric precision for the released 40k VC.

This is a decoder+HiFT timing benchmark with cached source tokens, target
conditioning and Ears scores. It does not include raw-audio tokenization,
Ears/CAMPPlus ranking, MP3 encoding, model load, or network latency. FP8
quantizes only supported Linear layers of the flow estimator via torchao;
the convolutional path remains BF16. Failed or zero-layer FP8 attempts are
recorded as unavailable rather than reported as an FP8 speed result.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import MethodType

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'grow_publish'))
sys.path.insert(0, str(PROJECT / 'grow_v1'))
sys.path.insert(0, str(PROJECT))
from generate_listening import ref_dict, set_adapter
from train_grow import conditioning_scores
from infer import load_converter

INPUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval')
RELEASE = Path('/e/scratch/laionize/schuhmann1/humaneness_voice_converter_release')
OUT = Path('/e/scratch/laionize/schuhmann1/hvc_speed_2026_10_11')
CASES = ('25-4', '00-4', '24-2')
SIZES = (1, 4, 8, 16)


def bench(model, source, target, names, mode, repeats, device):
    set_adapter(model, conditioning_scores(source, target, names), 1.0, device)
    reference = ref_dict(target, device, acoustic_prompt=True)
    codes = np.frombuffer(source['tokens'], dtype='<i2').astype(np.int64)
    token = torch.from_numpy(codes[None]).to(device)
    result = []
    for n in SIZES:
        # One pair, n independent initial CFM noise draws in the same forward.
        # The same fixed batch RNG seed makes repetitions reproducible, but
        # batch row i is not bit-identical to serial candidate i.
        samples = []
        peak_bytes = []
        durations = []
        for trial in range(repeats + 1):
            torch.manual_seed(20261011 + n)
            torch.cuda.manual_seed_all(20261011 + n)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            context = (torch.autocast('cuda', dtype=torch.bfloat16)
                       if mode != 'fp32' else nullcontext())
            # torchao 0.15 Float8Tensor requires a version counter during
            # transpose; inference_mode removes it. FP8 uses no_grad instead.
            grad_context = torch.no_grad() if mode == 'fp8' else torch.inference_mode()
            with grad_context, context:
                mel = model.flow_inference(speech_tokens=token.expand(n, -1).contiguous(),
                                           ref_dict=dict(reference),
                                           n_cfm_timesteps=10, finalize=True)
                wave, _ = model.hift_inference(mel)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            audio = wave.float().cpu().numpy()
            if audio.shape[0] != n or not np.isfinite(audio).all():
                raise AssertionError(f'{mode} n={n}: invalid audio shape/values {audio.shape}')
            if trial:
                samples.append(elapsed)
                peak_bytes.append(torch.cuda.max_memory_allocated(device))
                durations.append(audio.shape[-1] / 24000)
            del audio, wave, mel
        median = float(statistics.median(samples))
        duration = float(statistics.median(durations))
        row = {'case_id': source['case_id'], 'input_seconds': source['dur_s'],
               'target': target['target_label'], 'mode': mode, 'candidates': n,
               'wall_seconds_median': median, 'candidate_seconds_per_wall_second': n * duration / median,
               'winner_latency_rtf': median / duration,
               'per_candidate_amortized_seconds': median / n,
               'output_seconds': duration,
               'peak_allocated_gib': max(peak_bytes) / 2**30,
               'repetitions': repeats, 'warmups': 1,
               'all_wall_seconds': samples}
        print(json.dumps({'event': 'SPEED_CASE', **row}), flush=True)
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--mode', choices=('fp32', 'bf16', 'fp8'), required=True)
    args = parser.parse_args()
    assert args.repeats >= 2 and torch.cuda.is_available()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    model = load_converter(RELEASE, device)
    model.eval()
    fp8 = {'status': 'not_requested'}
    if args.mode == 'fp8':
        from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig,
                                           quantize_)
        from torchao.quantization.granularity import PerTensor
        estimator = model.flow.decoder.estimator
        linear_count = sum(isinstance(module, nn.Linear) for module in estimator.modules())
        fp8 = {'status': 'attempted', 'target': 'flow.decoder.estimator Linear layers',
               'linear_count': linear_count, 'other_layers': 'BF16 autocast',
               'torchao': '0.15.0'}
        if linear_count == 0:
            raise RuntimeError('No estimator Linear layers; cannot label this FP8')
        quantize_(estimator, Float8DynamicActivationFloat8WeightConfig(
            granularity=PerTensor()))
        # Upstream wraps both paths in inference_mode, incompatible with the
        # Float8Tensor weight transpose/version counter in torchao 0.15.
        # Remove only those wrappers for this experimental FP8 measurement;
        # keep global no_grad in bench(). This overhead must be disclosed.
        model.flow.inference = MethodType(type(model.flow).inference.__wrapped__, model.flow)
        model.flow_inference = MethodType(type(model).flow_inference.__wrapped__, model)
        model.flow.decoder.forward = MethodType(type(model.flow.decoder).forward.__wrapped__,
                                                model.flow.decoder)
        fp8['inference_mode_wrappers_removed'] = True
        fp8['status'] = 'quantized'
    prep = json.loads((INPUT / 'prepared.COMPLETE.json').read_text())
    rows = pq.read_table(INPUT / 'prepared.parquet').to_pylist()
    sources = {row['case_id']: row for row in rows if row['role'] == 'source'}
    target = next(row for row in rows if row['role'] == 'target' and row['target_label'] == 'B')
    measured = []
    started = time.perf_counter()
    for case_id in CASES:
        measured.extend(bench(model, sources[case_id], target, prep['score_order'],
                              args.mode, args.repeats, device))
    artifact = {'status': 'COMPLETE', 'mode': args.mode,
                'scope': 'precomputed tokens, reference, scores; full 10-step flow plus HiFT waveform',
                'exclusions': ['model loading', 'S3 tokenization', 'Ears/CAMPPlus ranking',
                               'MP3 encoding', 'network'],
                'candidate_sampling': 'independent batched Gaussian CFM noise; fixed batch seed',
                'gpu': torch.cuda.get_device_name(device),
                'compute_capability': torch.cuda.get_device_capability(device),
                'torch': torch.__version__, 'precision': fp8, 'rows': measured,
                'total_wall_seconds': time.perf_counter() - started}
    path = OUT / f'{args.mode}.json'
    temp = path.with_suffix('.json.partial')
    temp.write_text(json.dumps(artifact, indent=2) + '\n')
    os.replace(temp, path)
    print(json.dumps({'event': 'SPEED_DONE', 'path': str(path)}), flush=True)


if __name__ == '__main__':
    main()
