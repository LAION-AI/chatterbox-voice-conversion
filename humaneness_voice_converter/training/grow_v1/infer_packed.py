#!/usr/bin/env python3
"""Generate one VC example from two prepared S3Gen Parquet rows.

This is an inference entry point for the released 300-hour adapter and its
`continue` flow-RL snapshot. It intentionally does not depend on reward-model
weights. Input rows must already contain S3 tokens, mel, CAMPPlus and the 99
score dimensions in the published order; see the model card for preparation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from safetensors.torch import load_file
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_listening import load_model, ref_dict, set_adapter
from select_listening import TARGET_STATIC
from train import LoRALinear


def get_row(path: Path, uid: str) -> dict:
    # A row-group scan avoids materializing a potentially large container.
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if row['uid'] == uid:
                return row
    raise ValueError(f'UID not found: {uid}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True, help='Prepared Parquet container')
    parser.add_argument('--source-uid', required=True, help='Content/emotion source UID')
    parser.add_argument('--target-uid', required=True, help='5–10 s target-voice UID')
    parser.add_argument('--score-order-json', type=Path, required=True,
                        help='JSON list of 99 score names, or train-0.COMPLETE.json')
    parser.add_argument('--rl-checkpoint', type=Path,
                        help='Optional RL step checkpoint; omit for the 300 h adapter')
    parser.add_argument('--arm', choices=('continue', 'merged_new'), default='continue',
                        help='Required reconstruction branch for the RL checkpoint')
    parser.add_argument('--output-wav', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1901)
    parser.add_argument('--nfe', type=int, default=10)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'A CUDA GPU is required'
    raw_names = json.loads(args.score_order_json.read_text())
    names = raw_names['score_order'] if isinstance(raw_names, dict) else raw_names
    assert len(names) == 99 and TARGET_STATIC.issubset(set(names))
    source = get_row(args.pool, args.source_uid)
    target = get_row(args.pool, args.target_uid)
    source_scores = np.frombuffer(source['scores'], dtype='<f4').copy()
    target_scores = np.frombuffer(target['scores'], dtype='<f4')
    assert source_scores.shape == target_scores.shape == (99,)
    for name in TARGET_STATIC:
        source_scores[names.index(name)] = target_scores[names.index(name)]

    device = torch.device('cuda:0')
    model = load_model(device)
    if args.rl_checkpoint:
        if args.arm == 'merged_new':
            # Recreate the training arm's effective base: fold the original
            # supervised linear LoRA first, then load the fresh RL adapter.
            folded = 0
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, LoRALinear):
                        module.base.weight.add_((module.lora_up.weight @
                                                 module.lora_down.weight).to(module.base.weight))
                        folded += 1
            assert folded == 224
        if args.rl_checkpoint.suffix == '.safetensors':
            adapted = load_file(args.rl_checkpoint)
            rl_step = None
        else:
            # .pt is an optimizer-bearing pickle; load only trusted files.
            state = torch.load(args.rl_checkpoint, map_location='cpu', weights_only=False)
            assert state['schema'] == 'chatterbox-s3gen-grow-v1'
            assert state['arm'] == args.arm, 'Checkpoint arm does not match --arm'
            adapted = state['trainable']
            rl_step = state['step']
        trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
        assert set(trainable) == set(adapted)
        for name, param in trainable.items():
            param.data.copy_(adapted[name].to(device))
    model.eval()
    set_adapter(model, source_scores, 1.0, device)
    reference = ref_dict(target, device, acoustic_prompt=True)
    tokens = np.frombuffer(source['tokens'], dtype='<i2').astype(np.int64)
    token_tensor = torch.from_numpy(tokens[None]).to(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        mel = model.flow_inference(speech_tokens=token_tensor,
                                   ref_dict=dict(reference), n_cfm_timesteps=args.nfe,
                                   finalize=True)
        generated, _ = model.hift_inference(mel)
    wave = generated.float().cpu().numpy().reshape(-1)
    assert np.isfinite(wave).all() and len(wave) >= 24000
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, np.clip(wave, -1.0, 1.0), 24000)
    print(json.dumps({'output': str(args.output_wav), 'seconds': len(wave) / 24000,
                      'source_uid': args.source_uid, 'target_uid': args.target_uid,
                      'arm': args.arm if args.rl_checkpoint else 'supervised_base',
                      'rl_step': rl_step if args.rl_checkpoint else None}))


if __name__ == '__main__':
    main()
