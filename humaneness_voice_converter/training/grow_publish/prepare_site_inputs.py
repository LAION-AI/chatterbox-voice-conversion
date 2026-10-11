#!/usr/bin/env python3
"""Prepare the 160 S10 listening sources and two fixed target references.

This re-tokenizes the exact MP3 bytes heard on the supplied page, so the new
four-model comparison has identical source/target input audio in every arm.
The two target score vectors are taken from the existing 300-hour packed
training data by UID; only audio-derived tokens, mel and CAMPPlus are rebuilt.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.torch import load_file
import soundfile as sf
import torch
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3gen.utils.mel import mel_spectrogram

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare import WEIGHTS, ROOT, resample
from s3_tokenization import tokenize_waveform

OUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval')


def target_scores(uids: set[str]) -> dict[str, bytes]:
    found = {}
    for shard in range(4):
        path = ROOT / 'packed' / f'train-{shard}.parquet'
        for batch in pq.ParquetFile(path).iter_batches(columns=['uid', 'scores'], batch_size=1024):
            for row in batch.to_pylist():
                if row['uid'] in uids:
                    found[row['uid']] = row['scores']
    assert set(found) == uids, (set(found), uids)
    return found


def main() -> None:
    assert torch.cuda.is_available()
    proof = json.loads((OUT / 'site_inputs.COMPLETE.json').read_text())
    assert proof['status'] == 'COMPLETE' and proof['cases'] == 160
    manifest = json.loads((OUT / 'site_inputs.json').read_text())
    assert len(manifest['cases']) == 160 and len(manifest['targets']) == 2
    target_uids = {target['uid'] for target in manifest['targets']}
    t_scores = target_scores(target_uids)
    names = json.loads((ROOT / 'packed/train-0.COMPLETE.json').read_text())['score_order']
    assert len(names) == 99
    training_sources = set(pq.read_table(ROOT / 'grow_v1/selection.parquet',
                                         columns=['source_uid']).column('source_uid').to_pylist())
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    model = S3Gen()
    missing, unexpected = model.load_state_dict(load_file(WEIGHTS), strict=False)
    assert not unexpected and set(missing) <= {'tokenizer.window'}
    model.to(device).eval().requires_grad_(False)
    audio_tar = OUT / 'site_inputs.tar'
    records = []
    with tarfile.open(audio_tar, 'r') as archive:
        items = [(case, f'sources/{case["case_id"]}.mp3', 'source')
                 for case in manifest['cases']]
        items += [(target, f'references/{target["label"]}.mp3', 'target')
                  for target in manifest['targets']]
        for index, (meta, member, role) in enumerate(items):
            payload = archive.extractfile(member).read()
            assert hashlib.sha256(payload).hexdigest() == manifest['audio_sha256'][member]
            wave, sr = sf.read(io.BytesIO(payload), dtype='float32', always_2d=True)
            wave = wave.mean(axis=1)
            assert 2.0 <= len(wave) / sr <= 15.5 and np.isfinite(wave).all()
            w16 = resample(wave, sr, 16000)
            w24 = resample(wave, sr, 24000)
            with torch.inference_mode():
                token = tokenize_waveform(model.tokenizer, w16)
                assert len(token) >= 45 and token.max() < 6561
                camp = model.speaker_encoder.inference(
                    torch.from_numpy(w16[:10 * 16000].copy())[None].to(device)).flatten().float()
                mel = mel_spectrogram(torch.from_numpy(w24).to(device)[None]).squeeze(0).float()
            expected = 2 * len(token)
            assert camp.numel() == 192 and mel.shape[0] == 80
            assert abs(mel.shape[1] - expected) <= 10, (member, mel.shape, len(token))
            if mel.shape[1] < expected:
                mel = torch.nn.functional.pad(mel, (0, expected - mel.shape[1]), mode='replicate')
            mel = mel[:, :expected].contiguous()
            if role == 'source':
                scores = np.asarray(meta['scores_raw'], dtype=np.float32)
                assert scores.shape == (99,)
                scores[:98] /= 6.0
                scores[98] /= 10.0
            else:
                scores = np.frombuffer(t_scores[meta['uid']], dtype='<f4').copy()
            assert np.isfinite(scores).all()
            records.append({'uid': meta['uid'], 'role': role,
                            'case_id': meta.get('case_id'), 'target_label': meta.get('label'),
                            'emotion': meta.get('emotion'), 'lang': meta.get('lang'),
                            'origin': meta.get('origin'), 'text': meta.get('text'),
                            'dur_s': float(len(w16) / 16000),
                            'tokens': np.asarray(token, dtype='<i2').tobytes(),
                            'token_count': int(len(token)),
                            'mel': mel.cpu().numpy().astype('<f2').tobytes(),
                            'mel_frames': expected,
                            'camp': camp.cpu().numpy().astype('<f2').tobytes(),
                            'scores': scores.astype('<f4').tobytes(),
                            'record_path': str(audio_tar), 'audio_member': member,
                            'rl_training_uid_overlap': role == 'source' and meta['uid'] in training_sources})
            if (index + 1) % 20 == 0:
                print(json.dumps({'prepared': index + 1, 'total': len(items)}), flush=True)
    assert len(records) == 162
    target = OUT / 'prepared.parquet'
    temporary = target.with_suffix('.parquet.partial')
    pq.write_table(pa.Table.from_pylist(records), temporary,
                   compression='zstd', row_group_size=64)
    os.replace(temporary, target)
    check = {'status': 'COMPLETE', 'rows': 162, 'sources': 160, 'targets': 2,
             'rl_training_source_uid_overlap': sum(r['rl_training_uid_overlap'] for r in records),
             'score_order': names, 'raw_site_input_sha256': proof['tar_sha256'],
             'weights_sha256': hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
             'output': str(target), 'new_per_sample_files': 0}
    complete = OUT / 'prepared.COMPLETE.json'
    temporary = complete.with_suffix('.json.partial')
    temporary.write_text(json.dumps(check, indent=2) + '\n')
    os.replace(temporary, complete)
    print(json.dumps({k: v for k, v in check.items() if k != 'score_order'}), flush=True)


if __name__ == '__main__':
    main()
