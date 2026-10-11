"""One GPU per packed shard: Chatterbox tokens, mel, CAMPPlus and 99 scores."""
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import resample_poly
from safetensors.torch import load_file
import soundfile as sf
import torch

from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3gen.utils.mel import mel_spectrogram
from s3_tokenization import tokenize_waveform

SC = Path(os.environ.get('CHATTERBOX_SCRATCH', '/e/scratch/reformo/schuhmann1_moss'))
ROOT = Path(os.environ.get('CHATTERBOX_RUN_ROOT', str(SC / 'out/chatterbox_s3gen_300h_v1')))
WEIGHTS = Path(os.environ.get('CHATTERBOX_S3GEN_WEIGHTS', str(
    SC / 'hfcache/hub/models--ResembleAI--chatterbox/snapshots/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18/s3gen.safetensors')))

# The S6 sidecar uses display labels for eleven of the same 40 emotion axes.
# Map by explicit meaning, never by sorted position: position-based mixing
# would silently attach the wrong emotions to sidecar training examples.
EMO_ALIASES = {
    'Astonishment/Surprise': 'Astonishment_Surprise',
    'Emotional Numbness': 'Emotional_Numbness',
    'Fatigue/Exhaustion': 'Fatigue_Exhaustion',
    'Hope/Optimism': 'Hope_Enthusiasm_Optimism',
    'Impatience and Irritability': 'Impatience_and_Irritability',
    'Intoxication/Altered States': 'Intoxication_Altered_States_of_Consciousness',
    'Jealousy & Envy': 'Jealousy_and_Envy',
    'Malevolence/Malice': 'Malevolence_Malice',
    'Pleasure/Ecstasy': 'Pleasure_Ecstasy',
    'Sexual Lust': 'Sexual_Lust',
    'Thankfulness/Gratitude': 'Thankfulness_Gratitude',
}


def normalized_emotions(raw):
    normalized = {EMO_ALIASES.get(name, name): value for name, value in raw.items()}
    assert len(normalized) == len(raw) == 40, 'Emotion name collision or missing axis'
    return normalized


def resample(wave, src, dst):
    if src == dst:
        return wave.astype(np.float32, copy=False)
    common = np.gcd(src, dst)
    return resample_poly(wave, dst // common, src // common).astype(np.float32)


def main():
    heldout = os.environ.get('PREP_HOLDOUT') == '1'
    rank = int(os.environ['SLURM_PROCID'])
    world = int(os.environ['SLURM_NTASKS'])
    limit = int(os.environ.get('PREP_LIMIT', '0'))
    source_filter = os.environ.get('PREP_SOURCE')
    assert (rank == 0 and world in (1, 4)) if heldout else (world == 4 and 0 <= rank < 4) or (world == 1 and rank == 0 and limit > 0)
    assert not source_filter or (source_filter == 'sidecar' and limit > 0 and not heldout)
    torch.set_num_threads(4)
    device = torch.device('cuda:0' if torch.cuda.device_count() == 1 else 'cuda:' + os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(device)
    selected = json.loads((ROOT / 'SELECTION.json').read_text())
    assert selected['status'] == 'SELECTED' and 300 <= selected['train_hours'] <= 300.1
    assert limit >= 0 and not (heldout and limit)
    split_name = 'heldout' if heldout else f'train-{rank}'
    rows = pq.read_table(ROOT / f'{split_name}.parquet').to_pylist()
    if source_filter:
        rows = [row for row in rows if row['source'] == source_filter]
    if limit:
        rows = rows[:limit]
        split_name = f'canary-{source_filter}-{split_name}' if source_filter else f'canary-{split_name}'
    rows.sort(key=lambda r: (r['record_locator']['path'], r['uid']))
    model = S3Gen()
    missing, unexpected = model.load_state_dict(load_file(WEIGHTS), strict=False)
    assert not unexpected and set(missing) <= {'tokenizer.window'}, (missing, unexpected)
    model = model.to(device).eval()
    out_dir = ROOT / 'packed'
    out_dir.mkdir(exist_ok=True)
    target = out_dir / f'{split_name}.parquet'
    assert not target.exists() and not (out_dir / f'{split_name}.COMPLETE.json').exists()
    out = []
    rejected = []
    source_hours = collections.Counter()
    aliased_rows = 0
    current_tar = None
    archive = None
    started = time.monotonic()
    emo_names = vn_names = None
    for i, row in enumerate(rows):
        uid = row['uid']
        try:
            loc = row['record_locator']
            if current_tar != loc['path']:
                if archive is not None:
                    archive.close()
                archive = tarfile.open(loc['path'], 'r')
                current_tar = loc['path']
            meta = json.loads(archive.extractfile(loc['member']).read())
            assert meta['uid'] == uid
            payload = archive.extractfile(uid + '.mp3').read()
            wave, sr = sf.read(io.BytesIO(payload), dtype='float32')
            if wave.ndim == 2:
                wave = wave.mean(axis=1)
            assert np.isfinite(wave).all() and 2 <= len(wave) / sr <= 25.2
            w16 = resample(wave, sr, 16000)
            w24 = resample(wave, sr, 24000)
            with torch.inference_mode():
                s3 = tokenize_waveform(model.tokenizer, w16)
                # Training windows require at least 45 semantic tokens.
                assert len(s3) >= 45 and s3.max() < 6561
                # Follow Chatterbox VC's ten-second reference window.
                ref16 = torch.from_numpy(w16[:10 * 16000].copy()).to(device)
                camp = model.speaker_encoder.inference(ref16[None]).flatten().float()
                mel = mel_spectrogram(torch.from_numpy(w24).to(device)[None]).squeeze(0).float()
            assert camp.numel() == 192 and torch.isfinite(camp).all()
            assert mel.ndim == 2 and mel.shape[0] == 80 and torch.isfinite(mel).all()
            expected = 2 * len(s3)
            assert abs(mel.shape[1] - expected) <= 10, (uid, mel.shape, len(s3))
            if mel.shape[1] < expected:
                mel = torch.nn.functional.pad(mel, (0, expected - mel.shape[1]), mode='replicate')
            mel = mel[:, :expected].contiguous()
            scores = meta['scores']
            emotion_scores = normalized_emotions(scores['emo'])
            if emotion_scores.keys() != scores['emo'].keys():
                aliased_rows += 1
            emo = sorted(emotion_scores)
            vn = sorted(scores['dims'])
            assert len(emo) == 40 and len(vn) == 57
            if emo_names is None:
                emo_names, vn_names = emo, vn
            assert emo == emo_names and vn == vn_names
            vals = [emotion_scores[k] for k in emo] + [scores['dims'][k] for k in vn] + [scores['genu'], scores['blend']]
            assert all(v is not None and np.isfinite(v) for v in vals)
            vals = np.asarray(vals, dtype=np.float32)
            vals[:98] /= 6.0
            vals[98] /= 10.0
            assert len(vals) == 99
            out.append({'uid': uid, 'source': row['source'], 'lang': meta.get('lang'),
                        'dur_s': float(len(w16) / 16000),
                        'tokens': np.asarray(s3, dtype='<i2').tobytes(), 'token_count': len(s3),
                        'mel': mel.cpu().numpy().astype('<f2').tobytes(), 'mel_frames': expected,
                        'camp': camp.cpu().numpy().astype('<f2').tobytes(),
                        'scores': vals.astype('<f4').tobytes(),
                        'record_path': loc['path'], 'audio_member': uid + '.mp3'})
            source_hours[row['source']] += len(w16) / 16000 / 3600
        except Exception as exc:
            rejected.append({'uid': uid, 'error': type(exc).__name__ + ': ' + str(exc)[:200]})
            if len(rejected) > max(20, int(.01 * len(rows))):
                raise RuntimeError(f'Too many rejected rows: {len(rejected)}') from exc
        if (i + 1) % 1000 == 0:
            print(json.dumps({'rank': rank, 'done': i + 1, 'total': len(rows), 'accepted': len(out), 'rejected': len(rejected)}), flush=True)
    if archive is not None:
        archive.close()
    assert len(out) >= .99 * len(rows)
    tmp = target.with_suffix('.partial')
    pq.write_table(pa.Table.from_pylist(out), tmp, compression='zstd', row_group_size=256)
    os.replace(tmp, target)
    proof = {'status': 'COMPLETE', 'rank': rank, 'split': split_name, 'rows': len(out),
             'hours': sum(r['dur_s'] for r in out) / 3600,
             'source_hours': dict(source_hours), 'rejected': rejected[:100],
             'rejected_count': len(rejected), 'aliased_emotion_rows': aliased_rows,
             'score_order': emo_names + vn_names + ['genuineness_0_6', 'blend_0_10'],
             's3_rate': 25, 's3_vocab': 6561, 'mel_rate': 50, 'camp_dim': 192,
             'weights_sha256': hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
             'elapsed_s': time.monotonic() - started,
             'new_per_sample_files': 0}
    (out_dir / f'{split_name}.COMPLETE.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({k: v for k, v in proof.items() if k != 'score_order'}), flush=True)


if __name__ == '__main__':
    main()
