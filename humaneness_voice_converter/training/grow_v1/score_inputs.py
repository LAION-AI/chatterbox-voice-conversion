#!/usr/bin/env python3
"""Frozen Humaneness Ears Medium (Whisper Small) input scores, packed per GPU."""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.torch import load_file
from scipy.signal import resample_poly
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor

BASE = Path(os.environ.get('CHATTERBOX_RUN_ROOT',
    '/e/scratch/reformo/schuhmann1_moss/out/chatterbox_s3gen_300h_v1'))
OUT = BASE / 'grow_v1'
EARS = Path(os.environ.get('CHATTERBOX_EARS_MEDIUM', str(BASE / 'ears_medium')))
sys.path.insert(0, str(EARS))
from model import LayeredMultiTaskWhisper


def audio_bytes(archive: tarfile.TarFile, member: str) -> tuple[np.ndarray, int]:
    payload = archive.extractfile(member)
    assert payload is not None
    wave, rate = sf.read(io.BytesIO(payload.read()), dtype='float32', always_2d=True)
    wave = wave.mean(axis=1)
    assert .1 < len(wave) / rate <= 30 and np.isfinite(wave).all()
    return wave, rate


def to16(wave: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16000:
        return wave
    common = math.gcd(rate, 16000)
    return resample_poly(wave, 16000 // common, rate // common).astype(np.float32)


def main() -> None:
    rank, world = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS'])
    assert world == 4 and 0 <= rank < world and torch.cuda.is_available()
    torch.set_num_threads(4)
    device = torch.device('cuda:0' if torch.cuda.device_count() == 1 else 'cuda:' + os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(device)
    selection = json.loads((OUT / 'SELECTION.json').read_text())
    proof = json.loads((OUT / 'POOL_COMPLETE.json').read_text())
    assert selection['status'] == 'SELECTED' and proof['status'] == 'COMPLETE'
    source_uids = sorted({r['source_uid'] for r in pq.read_table(
        OUT / 'selection.parquet', columns=['source_uid']).to_pylist()})
    assigned = set(source_uids[rank::world])
    pool = [r for r in pq.read_table(OUT / 'pool.parquet').to_pylist() if r['uid'] in assigned]
    assert len(pool) == len(assigned)
    pool.sort(key=lambda r: (r['record_path'], r['audio_member']))
    norm = json.loads((EARS / 'training_normalization.json').read_text())
    names = norm['score_names']
    assert names[:40] == ['emo_' + name for name in selection['emotion_axis_order']]
    pq_index = names.index('audiobox:PQ')
    classes = json.loads((EARS / 'classes.json').read_text())['names']
    model = LayeredMultiTaskWhisper(EARS / 'medium', len(classes), initialize_pretrained=False)
    model.load_state_dict(load_file(EARS / 'medium/model.safetensors', device='cpu'), strict=True)
    model.to(device).eval().requires_grad_(False)
    extractor = WhisperFeatureExtractor.from_pretrained(EARS / 'medium', local_files_only=True)
    means = np.asarray(norm['score_mean'], np.float32)
    stds = np.asarray(norm['score_std'], np.float32)
    assert len(means) == len(stds) == len(names)
    started = time.monotonic()
    archive, current_path = None, None
    out, batch = [], []

    def flush() -> None:
        if not batch:
            return
        waves = [item[1] for item in batch]
        features = extractor(waves, sampling_rate=16000, padding='longest', truncation=True,
                             max_length=480000, return_attention_mask=True, return_tensors='np')
        mel = features['input_features'].astype(np.float32)
        mask = features['attention_mask'].astype(np.int64)
        if mel.shape[-1] % 2:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, 1)))
            mask = np.pad(mask, ((0, 0), (0, 1)))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(torch.from_numpy(mel).to(device), torch.from_numpy(mask).to(device),
                           predict_events=False)
        raw = result['scores'].float().cpu().numpy() * stds + means
        assert raw.shape[0] == len(batch) and np.isfinite(raw).all()
        for (row, _wave), values in zip(batch, raw):
            out.append({'uid': row['uid'], 'ears_emotions_raw_f32':
                        values[:40].astype('<f4').tobytes(),
                        'ears_pq_raw': float(values[pq_index]),
                        'audio_duration_s': len(_wave) / 16000})
        batch.clear()

    for index, row in enumerate(pool):
        if row['record_path'] != current_path:
            flush()
            if archive is not None:
                archive.close()
            archive = tarfile.open(row['record_path'], 'r')
            current_path = row['record_path']
        wave, rate = audio_bytes(archive, row['audio_member'])
        batch.append((row, to16(wave, rate)))
        if len(batch) == 8:
            flush()
        if (index + 1) % 200 == 0:
            print(json.dumps({'rank': rank, 'done': index+1, 'total': len(pool),
                              'elapsed_s': time.monotonic()-started}), flush=True)
    flush()
    if archive is not None:
        archive.close()
    assert len(out) == len(pool)
    path = OUT / f'input_ears_{rank}.parquet'
    assert not path.exists()
    temporary = path.with_suffix('.parquet.partial')
    pq.write_table(pa.Table.from_pylist(out), temporary, compression='zstd')
    os.replace(temporary, path)
    complete = {'status': 'COMPLETE', 'rank': rank, 'rows': len(out),
                'model': 'laion/humaneness-ears-base-medium medium = Whisper Small',
                'revision': '818506970d93c809a3295a0a02c40dee8ff2bfc3',
                'score_order': names[:40], 'audiobox_axis': 'audiobox:PQ',
                'elapsed_s': time.monotonic()-started}
    temporary = OUT / f'input_ears_{rank}.COMPLETE.json.partial'
    temporary.write_text(json.dumps(complete, indent=2) + '\n')
    os.replace(temporary, OUT / f'input_ears_{rank}.COMPLETE.json')
    print(json.dumps({k: v for k, v in complete.items() if k != 'score_order'}), flush=True)


if __name__ == '__main__':
    main()
