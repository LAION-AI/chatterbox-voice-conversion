#!/usr/bin/env python3
"""Export a self-contained S3Gen+score-projector SafeTensors inference model.

The final optimizer checkpoint holds every trainable flow parameter. This
reconstructs its exact merged 24k parent, installs the 40k flow parameters,
and adds the frozen Chatterbox tokenizer/speaker encoder/HiFT weights. It is
run on a compute node, not a login node.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from safetensors.torch import save_file
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'grow_v4'))
from train_full import OUT, STEPS, load_merged_model

DEST = Path(os.environ.get('HVC_RELEASE_STAGE',
    '/e/scratch/laionize/schuhmann1/humaneness_voice_converter_release'))


def main() -> None:
    assert torch.cuda.is_available(), 'Run this export on a GPU compute node'
    done = json.loads((OUT / 'DONE.json').read_text())
    assert done['status'] == 'COMPLETE' and done['step'] == STEPS == 40000
    checkpoint = OUT / 'checkpoints/step-0040000.pt'
    proof = json.loads(checkpoint.with_suffix('.COMPLETE.json').read_text())
    assert proof['status'] == 'COMPLETE' and checkpoint.stat().st_size == proof['bytes']
    device = torch.device('cuda:0')
    model, merge_error, flow_params = load_merged_model(device)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['schema'] == 'chatterbox-s3gen-grow-v4-full-403030'
    assert state['step'] == 40000 and state['group_size'] == 16
    names = {name: parameter for name, parameter in model.named_parameters()
             if parameter.requires_grad}
    assert names.keys() == state['trainable'].keys()
    with torch.no_grad():
        for name, parameter in names.items():
            parameter.copy_(state['trainable'][name].to(device))
    del state
    model.eval().requires_grad_(False)
    tensors = {key: value.detach().cpu().contiguous()
               for key, value in model.state_dict().items()}
    assert not any('lora_' in key for key in tensors)
    assert any(key.startswith('tokenizer.') for key in tensors)
    # Chatterbox names its HiFT vocoder `mel2wav` in the S3Gen state dict.
    assert any(key.startswith('mel2wav.') for key in tensors)
    assert any('.spk_embed_affine_layer.style.' in key for key in tensors)
    DEST.mkdir(parents=True, exist_ok=True)
    output = DEST / 'model.safetensors'
    partial = output.with_suffix('.safetensors.partial')
    save_file(tensors, partial, metadata={
        'format': 'pt', 'architecture': 'Chatterbox S3Gen with LAION 99-score conditioning',
        'training_step': '40000', 'base_model': 'ResembleAI/chatterbox',
        'base_revision': '5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18',
    })
    os.replace(partial, output)
    digest = hashlib.sha256()
    with output.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    summary = {
        'status': 'COMPLETE', 'file': 'model.safetensors',
        'sha256': digest.hexdigest(),
        'bytes': output.stat().st_size, 'tensor_count': len(tensors),
        'flow_trainable_parameters': flow_params,
        'parent_lora_merge_max_abs_error': merge_error,
        'source_optimizer_checkpoint': str(checkpoint),
        'upstream_s3gen_revision': '5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18',
    }
    (DEST / 'EXPORT_PROOF.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
