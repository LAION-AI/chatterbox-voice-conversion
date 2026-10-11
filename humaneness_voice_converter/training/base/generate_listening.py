"""Matched-seed listening ablations; 40 emotions + 40 cross-speaker VC cases.

All MP3 audio is written into one TAR per GPU. No per-sample files are made.
This exercises S3Gen only: no T3 text generation, ASR or voice-quality claims.
"""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.torch import load_file
import torch
from torch import nn

from chatterbox.models.s3gen import S3Gen
from train import ROOT, WEIGHTS, LoRALinear, StyledSpeakerProjection
from select_listening import OUT, TARGET_STATIC

FFMPEG = os.environ.get('CHATTERBOX_FFMPEG',
    '/e/scratch/reformo/schuhmann1_moss/code/vc_env_pkgs/imageio_ffmpeg/binaries/ffmpeg-linux-aarch64-v7.0.2')
CKPT = Path(os.environ.get('CHATTERBOX_BASE_ADAPTER', str(ROOT / 'checkpoints/step-0021350.pt')))


def encode_mp3(wave):
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    assert len(wave) >= 24000 and np.isfinite(wave).all()
    clip_fraction = float(np.mean(np.abs(wave) > 1.0))
    wave = np.clip(wave, -1.0, 1.0)
    cmd = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-f', 'f32le',
           '-ar', '24000', '-ac', '1', '-i', 'pipe:0', '-b:a', '128k',
           '-ar', '24000', '-ac', '1', '-f', 'mp3', 'pipe:1']
    process = subprocess.run(cmd, input=wave.tobytes(), capture_output=True,
                             timeout=120, check=True)
    assert process.stdout.startswith(b'ID3') or process.stdout[:2] == b'\xff\xfb'
    return process.stdout, clip_fraction


def reference_mp3(row):
    with tarfile.open(row['record_path'], 'r') as archive:
        original = archive.extractfile(row['audio_member']).read()
    cmd = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
           '-t', '10', '-b:a', '128k', '-ar', '24000', '-ac', '1',
           '-f', 'mp3', 'pipe:1']
    return subprocess.run(cmd, input=original, capture_output=True,
                          timeout=120, check=True).stdout


def load_model(device):
    model = S3Gen()
    missing, unexpected = model.load_state_dict(load_file(WEIGHTS), strict=False)
    assert not unexpected and set(missing) <= {'tokenizer.window'}, (missing, unexpected)
    model.requires_grad_(False)
    estimator = model.flow.decoder.estimator
    names = [name for name, module in estimator.named_modules()
             if name.endswith(('.attn1.to_q', '.attn1.to_k', '.attn1.to_v', '.attn1.to_out.0'))]
    assert len(names) == 224
    for name in names:
        parent_name, leaf = name.rsplit('.', 1)
        parent = estimator.get_submodule(parent_name)
        base = estimator.get_submodule(name)
        assert isinstance(base, nn.Linear)
        if isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(leaf)] = LoRALinear(base)
        else:
            setattr(parent, leaf, LoRALinear(base))
    model.flow.spk_embed_affine_layer = StyledSpeakerProjection(model.flow.spk_embed_affine_layer)
    if CKPT.suffix == '.safetensors':
        trainable = load_file(CKPT)
    else:
        # The .pt checkpoint includes optimizer/RNG state. Only load trusted files.
        trainable = torch.load(CKPT, map_location='cpu', weights_only=False)['trainable']
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    assert expected == set(trainable), (len(expected), len(trainable))
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data.copy_(trainable[name])
    model = model.to(device).eval()
    model.flow.decoder.training_cfg_rate = 0.0
    return model


def ref_dict(row, device, acoustic_prompt):
    camp = np.frombuffer(row['camp'], dtype='<f2').astype(np.float32)
    assert camp.shape == (192,)
    if acoustic_prompt:
        n = min(int(row['token_count']), 250)
        tokens = np.frombuffer(row['tokens'], dtype='<i2')[:n].astype(np.int64)
        mel = np.frombuffer(row['mel'], dtype='<f2').reshape(80, 2 * row['token_count'])[:, :2*n].T.copy().astype(np.float32)
    else:
        n = 0
        tokens = np.empty(0, dtype=np.int64)
        mel = np.empty((0, 80), dtype=np.float32)
    return {'prompt_token': torch.from_numpy(tokens[None]).to(device),
            'prompt_token_len': torch.tensor([n], device=device),
            'prompt_feat': torch.from_numpy(mel[None]).to(device),
            'prompt_feat_len': torch.tensor([2*n], device=device),
            'embedding': torch.from_numpy(camp[None]).to(device)}


def set_adapter(model, scores, scale, device):
    projector = model.flow.spk_embed_affine_layer
    projector.condition = torch.from_numpy(np.asarray(scores, dtype=np.float32)[None]).to(device)
    projector.present = torch.ones((1, 99), device=device)
    for module in model.modules():
        if isinstance(module, (LoRALinear, StyledSpeakerProjection)):
            module.scale = scale


def synthesize(model, source, ref, scores, scale, seed, device):
    tokens = np.frombuffer(source['tokens'], dtype='<i2').astype(np.int64)
    token = torch.from_numpy(tokens[None]).to(device)
    set_adapter(model, scores, scale, device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        audio, _ = model.inference(speech_tokens=token,
                                   ref_dict=dict(ref), n_cfm_timesteps=10)
    wave = audio.squeeze(0).float().cpu().numpy()
    assert np.isfinite(wave).all()
    return wave


def put(archive, member, payload):
    info = tarfile.TarInfo(member)
    info.size = len(payload)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    rank = int(os.environ['SLURM_PROCID'])
    world = int(os.environ['SLURM_NTASKS'])
    assert (world, rank) == (1, 0) if args.smoke else (world == 4 and 0 <= rank < 4)
    device = torch.device('cuda:0' if torch.cuda.device_count() == 1 else 'cuda:' + os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    selection = json.loads((OUT / 'SELECTION.json').read_text())
    assert selection['count'] == 40
    names = selection['score_order']
    rows = pq.read_table(OUT / 'selection.parquet').to_pylist()
    assert len(rows) == 40
    model = load_model(device)
    cases = [rows[0]] if args.smoke else rows[rank::world]
    prefix = 'smoke' if args.smoke else f'rank-{rank}'
    final_tar = OUT / f'{prefix}.tar'
    final_parquet = OUT / f'{prefix}.parquet'
    assert not final_tar.exists() and not final_parquet.exists()
    tmp_tar = OUT / f'{prefix}.tar.partial'
    records = []
    started = time.monotonic()
    with tarfile.open(tmp_tar, 'w') as archive:
        for case in cases:
            idx = case['emotion_index']
            source = case['source']
            target = case['target']
            source_scores = np.frombuffer(source['scores'], dtype='<f4').copy()
            target_scores = np.frombuffer(target['scores'], dtype='<f4').copy()
            assert source_scores.shape == target_scores.shape == (99,)
            mixed = source_scores.copy()
            for axis in TARGET_STATIC:
                column = names.index(axis)
                mixed[column] = target_scores[column]
            # Make the requested emotion unambiguously dominant for the
            # 40-axis conditioning probe; the VC arm keeps raw source style.
            focused = source_scores.copy()
            focused[:40] *= .5
            focused[idx] = max(.8, float(source_scores[idx]))
            neutral = focused.copy()
            neutral[:40] = 0.0
            content = ref_dict(source, device, acoustic_prompt=False)
            vc_ref = ref_dict(target, device, acoustic_prompt=True)
            ref_payloads = {'source': reference_mp3(source), 'target': reference_mp3(target)}
            for label, payload in ref_payloads.items():
                member = f'{idx:02d}/{label}.mp3'
                put(archive, member, payload)
                records.append({'emotion_index': idx, 'mode': 'reference', 'variant': label,
                                'member': member, 'duration_s': None, 'clip_fraction': None})
            experiments = [
                ('emotion', 'base', focused, 0., content),
                ('emotion', 'adapted', focused, 1., content),
                ('emotion', 'adapted_no_emotion_scores', neutral, 1., content),
                ('vc', 'base_target_voice', mixed, 0., vc_ref),
                ('vc', 'adapted_mixed', mixed, 1., vc_ref),
                ('vc', 'adapted_target_all_scores', target_scores, 1., vc_ref),
            ]
            for mode, variant, scores, scale, reference in experiments:
                seed = 20261009 + idx
                wave = synthesize(model, source, reference, scores, scale, seed, device)
                payload, clipped = encode_mp3(wave)
                member = f'{idx:02d}/{mode}_{variant}.mp3'
                put(archive, member, payload)
                records.append({'emotion_index': idx, 'mode': mode, 'variant': variant,
                                'member': member, 'duration_s': len(wave) / 24000,
                                'clip_fraction': clipped})
            print(json.dumps({'rank': rank, 'done_case': idx, 'emotion': case['emotion'],
                              'elapsed_s': round(time.monotonic() - started, 1)}), flush=True)
    os.replace(tmp_tar, final_tar)
    pq.write_table(pa.Table.from_pylist(records), final_parquet, compression='zstd')
    proof = {'status': 'COMPLETE', 'rank': rank, 'smoke': args.smoke,
             'cases': len(cases), 'audio_members': len(records),
             'tar': str(final_tar), 'parquet': str(final_parquet),
             'elapsed_s': time.monotonic() - started, 'new_loose_audio_files': 0}
    (OUT / f'{prefix}.COMPLETE.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
