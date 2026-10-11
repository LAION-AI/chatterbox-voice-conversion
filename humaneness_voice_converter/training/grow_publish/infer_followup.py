#!/usr/bin/env python3
"""Inference for the three 24k-group Chatterbox S3Gen VC follow-up arms.

Requires two prepared Parquet rows, not arbitrary WAV input. For the
remerged_new arm, pass the v1 merged_new step-4000 adapter as --parent-v1.
The parent is folded into the effective S3Gen base twice (supervised,
then v1 RL) before the new follow-up adapter is loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open
from safetensors.torch import load_file
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_listening import load_model, ref_dict, set_adapter
from select_listening import TARGET_STATIC
from train import LoRALinear

SCHEMAS = {
    'continue': 'chatterbox-s3gen-grow-v2-ddp',
    'remerged_new': 'chatterbox-s3gen-grow-v2-ddp',
    'speaker_mix': 'chatterbox-s3gen-grow-v3-ce-emo-speaker',
}


def get_row(path: Path, uid: str) -> dict:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if row['uid'] == uid:
                return row
    raise ValueError(f'UID not found: {uid}')


def load_trainables(path: Path, expected_schema: str, expected_arm: str) -> dict:
    if path.suffix == '.safetensors':
        with safe_open(str(path), framework='pt', device='cpu') as file:
            metadata = file.metadata() or {}
            assert metadata.get('arm') == expected_arm, metadata
        return load_file(str(path))
    # Trusted checkpoint only: full .pt files contain optimizer and RNG state.
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['schema'] == expected_schema and state['arm'] == expected_arm
    return state['trainable']


def install_trainables(model, tensors: dict, device: torch.device) -> None:
    target = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert set(target) == set(tensors), (len(target), len(tensors))
    for name, param in target.items():
        param.data.copy_(tensors[name].to(device))


def fold_linear_lora(model) -> None:
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.base.weight.add_((module.lora_up.weight @
                                         module.lora_down.weight).to(module.base.weight))
                count += 1
    assert count == 224


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--score-order-json', type=Path, required=True)
    parser.add_argument('--source-uid', required=True)
    parser.add_argument('--target-uid', required=True)
    parser.add_argument('--arm', choices=tuple(SCHEMAS), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--parent-v1', type=Path,
                        help='Required v1 merged_new step-4000 adapter for remerged_new')
    parser.add_argument('--output-wav', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1901)
    parser.add_argument('--nfe', type=int, default=10)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'A CUDA GPU is required'
    assert (args.parent_v1 is not None) == (args.arm == 'remerged_new')
    raw_names = json.loads(args.score_order_json.read_text())
    names = raw_names['score_order'] if isinstance(raw_names, dict) else raw_names
    assert len(names) == 99 and TARGET_STATIC.issubset(set(names))
    source, target = get_row(args.pool, args.source_uid), get_row(args.pool, args.target_uid)
    scores = np.frombuffer(source['scores'], dtype='<f4').copy()
    target_scores = np.frombuffer(target['scores'], dtype='<f4')
    assert scores.shape == target_scores.shape == (99,)
    for name in TARGET_STATIC:
        scores[names.index(name)] = target_scores[names.index(name)]
    device = torch.device('cuda:0')
    model = load_model(device)  # SFT 300-hour rank-128 adapter
    if args.arm == 'remerged_new':
        fold_linear_lora(model)  # original supervised adapter into base
        parent = load_trainables(args.parent_v1,
                                 'chatterbox-s3gen-grow-v1', 'merged_new')
        install_trainables(model, parent, device)
        fold_linear_lora(model)  # v1 merged_new RL adapter into effective base
    followup = load_trainables(args.checkpoint, SCHEMAS[args.arm], args.arm)
    install_trainables(model, followup, device)
    model.eval()
    set_adapter(model, scores, 1.0, device)
    reference = ref_dict(target, device, acoustic_prompt=True)
    token = np.frombuffer(source['tokens'], dtype='<i2').astype(np.int64)
    token = torch.from_numpy(token[None]).to(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        mel = model.flow_inference(speech_tokens=token, ref_dict=dict(reference),
                                   n_cfm_timesteps=args.nfe, finalize=True)
        generated, _ = model.hift_inference(mel)
    wave = generated.float().cpu().numpy().reshape(-1)
    assert np.isfinite(wave).all() and len(wave) >= 24000
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, np.clip(wave, -1.0, 1.0), 24000)
    print(json.dumps({'output': str(args.output_wav), 'arm': args.arm,
                      'seconds': len(wave) / 24000,
                      'source_uid': args.source_uid, 'target_uid': args.target_uid}))


if __name__ == '__main__':
    main()
