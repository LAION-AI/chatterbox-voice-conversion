#!/usr/bin/env python3
"""Matched-seed, heldout-source evaluation for both GROW arms and base adapter.

Writes one TAR of MP3 audio, one Parquet of metrics, and a compact local
listening HTML. This is a comparison, not a claim that reward-model scores are
independent evidence of subjective quality.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path
import sys
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from generate_listening import encode_mp3, load_model, ref_dict, set_adapter
from score_inputs import audio_bytes, to16
from train_grow import (BASE, OUT, TRAIN_ROOT, EarsReward, conditioning_scores,
                        merge_and_reseed)

PREVIEW = Path(__file__).resolve().parent / 'GROW_REWARD_COMPARE_2026-10-09.html'
ARMS = ('baseline', 'continue', 'merged_new')
SEEDS = (1901, 1902, 1903)


def camp_vector(model, wave24: np.ndarray, device: torch.device) -> np.ndarray:
    wave16 = to16(np.asarray(wave24, dtype=np.float32), 24000)
    with torch.inference_mode():
        vec = model.speaker_encoder.inference(
            torch.from_numpy(wave16[None]).to(device)).float().cpu().numpy()[0]
    return vec / max(1e-8, float(np.linalg.norm(vec)))


def read_wave(row: dict) -> np.ndarray:
    with tarfile.open(row['record_path'], 'r') as archive:
        wave, rate = audio_bytes(archive, row['audio_member'])
    if rate != 24000:
        common = math.gcd(rate, 24000)
        wave = resample_poly(wave, 24000 // common, rate // common).astype(np.float32)
    return wave


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    member.mtime = 0
    archive.addfile(member, io.BytesIO(data))


def selection(rows: list[dict], names: list[str]) -> list[dict]:
    candidates = [row for row in rows if 3 <= float(row['dur_s']) <= 8]
    targets = [row for row in rows if 5 <= float(row['dur_s']) <= 10]
    assert len(candidates) >= 100 and len(targets) >= 100
    score = lambda row, axis: float(np.frombuffer(row['scores'], dtype='<f4')[axis])
    used_sources, used_targets = set(), set()
    chosen = {}
    axes = sorted(range(40), key=lambda axis: max(score(row, axis) for row in candidates))
    for axis in axes:
        source = next(row for row in sorted(candidates, key=lambda r: -score(r, axis))
                      if row['uid'] not in used_sources)
        used_sources.add(source['uid'])
        src_vec = np.frombuffer(source['camp'], dtype='<f2').astype(np.float32)
        src_vec /= max(1e-8, float(np.linalg.norm(src_vec)))
        options = []
        for target in targets:
            if target['uid'] == source['uid'] or target['uid'] in used_targets:
                continue
            tgt_vec = np.frombuffer(target['camp'], dtype='<f2').astype(np.float32)
            tgt_vec /= max(1e-8, float(np.linalg.norm(tgt_vec)))
            cosine = float(src_vec @ tgt_vec)
            if cosine < .30:
                options.append((cosine, target['uid'], target))
        assert options, (axis, source['uid'])
        _, _, target = min(options, key=lambda item: (abs(item[0] - .1), item[1]))
        used_targets.add(target['uid'])
        chosen[axis] = {'axis': axis, 'emotion': names[axis], 'source': source,
                        'target': target, 'annotated_emotion_raw': 6 * score(source, axis)}
    assert len(chosen) == 40 and len(used_sources) == len(used_targets) == 40
    return [chosen[axis] for axis in range(40)]


def load_arm(arm: str, device: torch.device, interim: bool = False):
    model = load_model(device)
    if arm == 'merged_new':
        merge_and_reseed(model)
    if arm != 'baseline':
        if interim:
            # Only committed, checksum-verified optimizer checkpoints are used.
            step = {'continue': 750, 'merged_new': 500}[arm]
            checkpoint = TRAIN_ROOT / arm / 'checkpoints' / f'step-{step:07d}.pt'
        else:
            done = json.loads((TRAIN_ROOT / arm / 'DONE.json').read_text())
            assert done['status'] == 'COMPLETE' and done['step'] == 4000
            checkpoint = Path(done['final_checkpoint'])
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        target = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert state['schema'] == 'chatterbox-s3gen-grow-v1' and state['arm'] == arm
        if interim:
            assert state['step'] == step
        assert set(target) == set(state['trainable'])
        for name, parameter in target.items():
            parameter.data.copy_(state['trainable'][name].to(device))
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--interim', action='store_true',
                        help='20 held-out cases, one seed, committed step-750/500 checkpoints')
    args = parser.parse_args()
    seeds = SEEDS[:1] if args.interim else SEEDS
    preview = (HERE / 'GROW_INTERIM_STEP_0750_0500_LISTENING_2026-10-09.html'
               if args.interim else PREVIEW)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    scorer = EarsReward(device)
    names = json.loads((BASE / 'packed/train-0.COMPLETE.json').read_text())['score_order']
    heldout = pq.read_table(BASE / 'packed/heldout.parquet').to_pylist()
    cases = selection(heldout, names)
    if args.interim:
        cases = cases[:20]
    training_uids = {row['source_uid'] for row in pq.read_table(
        OUT / 'selection.parquet', columns=['source_uid']).to_pylist()}
    assert not training_uids.intersection(case['source']['uid'] for case in cases)
    raw_sources = [read_wave(case['source']) for case in cases]
    raw_targets = [read_wave(case['target']) for case in cases]
    source_scores = [scorer.score(wave[None])[0, :40] for wave in raw_sources]
    top3 = [np.argsort(score)[-3:][::-1] for score in source_scores]
    target_camp = []
    for case in cases:
        vector = np.frombuffer(case['target']['camp'], dtype='<f2').astype(np.float32)
        target_camp.append(vector / max(1e-8, float(np.linalg.norm(vector))))
    eval_dir = TRAIN_ROOT.parent / ('evaluation_interim' if args.interim else 'evaluation')
    eval_dir.mkdir(parents=True, exist_ok=True)
    audio_tmp = eval_dir / 'audio.tar.partial'
    records = []
    with tarfile.open(audio_tmp, 'w') as archive:
        for axis in range(40):
            for label, wave in (('source', raw_sources[axis]), ('target', raw_targets[axis])):
                mp3, _ = encode_mp3(wave)
                add_bytes(archive, f'references/{axis:02d}-{label}.mp3', mp3)
        for arm in ARMS:
            model = load_arm(arm, device, interim=args.interim)
            for axis, case in enumerate(cases):
                source, target = case['source'], case['target']
                scores = conditioning_scores(source, target, names)
                set_adapter(model, scores, 1.0, device)
                reference = ref_dict(target, device, acoustic_prompt=True)
                token = np.frombuffer(source['tokens'], dtype='<i2').astype(np.int64)
                token_tensor = torch.from_numpy(token[None]).to(device)
                for seed in seeds:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        mel = model.flow_inference(speech_tokens=token_tensor,
                                                   ref_dict=dict(reference),
                                                   n_cfm_timesteps=10, finalize=True)
                        generated, _ = model.hift_inference(mel)
                    wave = generated.float().cpu().numpy().reshape(-1)
                    assert np.isfinite(wave).all() and len(wave) >= 24000
                    values = scorer.score(wave[None])[0]
                    top = top3[axis]
                    mse = float(np.mean((values[top] - source_scores[axis][top]) ** 2))
                    speaker = float(camp_vector(model, wave, device) @ target_camp[axis])
                    mp3, clipped = encode_mp3(wave)
                    member = f'outputs/{arm}/{axis:02d}-seed-{seed}.mp3'
                    add_bytes(archive, member, mp3)
                    records.append({'arm': arm, 'axis': axis, 'emotion': case['emotion'],
                                    'seed': seed, 'source_uid': source['uid'],
                                    'target_uid': target['uid'], 'audiobox_pq_raw': float(values[scorer.pq_index]),
                                    'top3_emotion_mse': mse, 'target_camp_cosine': speaker,
                                    'audio_s': len(wave) / 24000,
                                    'clipped_fraction': clipped, 'mp3_member': member,
                                    'top3_source_ears': [scorer.names[int(i)] for i in top],
                                    'top3_source_raw': [float(source_scores[axis][int(i)]) for i in top]})
                print(json.dumps({'arm': arm, 'axis': axis, 'emotion': case['emotion']}), flush=True)
            del model
            torch.cuda.empty_cache()
    assert len(records) == len(cases) * len(ARMS) * len(seeds)
    audio_tmp.replace(eval_dir / 'audio.tar')
    table_tmp = eval_dir / 'metrics.parquet.partial'
    pq.write_table(pa.Table.from_pylist(records), table_tmp, compression='zstd')
    table_tmp.replace(eval_dir / 'metrics.parquet')
    averages = {arm: {key: float(np.mean([row[key] for row in records if row['arm'] == arm]))
                      for key in ('audiobox_pq_raw', 'top3_emotion_mse', 'target_camp_cosine')}
                for arm in ARMS}
    summary = {'status': 'COMPLETE', 'interim': args.interim,
               'checkpoint_steps': {'continue': 750, 'merged_new': 500} if args.interim else
                                   {'continue': 4000, 'merged_new': 4000},
               'cases': len(cases), 'seeds': list(seeds), 'arms': list(ARMS),
               'source_split': 'UID-disjoint heldout', 'source_overlap_with_rl_train': 0,
               'averages': averages, 'caveats': ['Ears is the training reward model, not an independent quality measure.',
                                                'CAMPPlus similarity is a proxy, not a listening test.',
                                                'Heldout is UID-disjoint, not guaranteed speaker-disjoint.']}
    (eval_dir / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    # Keep the listening page small: first seed for 20 axes, all three arms.
    import html
    with tarfile.open(eval_dir / 'audio.tar', 'r') as archive:
        player = lambda member: '<audio controls preload="none" src="data:audio/mpeg;base64,' + base64.b64encode(
            archive.extractfile(member).read()).decode() + '"></audio>'
        cards = []
        first_seed = seeds[0]
        lookup = {(row['axis'], row['arm'], row['seed']): row for row in records}
        for case in cases[:20]:
            axis = case['axis']
            rows = [f'<h3>{html.escape(case["emotion"])} · held-out source</h3>',
                    '<p>Original emotion source ' + player(f'references/{axis:02d}-source.mp3') +
                    ' · target voice ' + player(f'references/{axis:02d}-target.mp3') + '</p>']
            for arm in ARMS:
                metric = lookup[(axis, arm, first_seed)]
                rows.append(f'<p><b>{arm}</b> · PQ {metric["audiobox_pq_raw"]:.2f} · '
                            f'top-3 emotion MSE {metric["top3_emotion_mse"]:.3f} · '
                            f'target CAMP cosine {metric["target_camp_cosine"]:.3f}<br>' +
                            player(metric['mp3_member']) + '</p>')
            cards.append('<section>' + ''.join(rows) + '</section>')
    avg_rows = ''.join(f'<tr><td>{arm}</td><td>{averages[arm]["audiobox_pq_raw"]:.3f}</td>'
                       f'<td>{averages[arm]["top3_emotion_mse"]:.3f}</td>'
                       f'<td>{averages[arm]["target_camp_cosine"]:.3f}</td></tr>' for arm in ARMS)
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Chatterbox GROW comparison</title>'
                '<style>body{max-width:1100px;margin:auto;font:16px system-ui;padding:24px;color:#172334}'
                'section{border:1px solid #ccd8e1;padding:16px;margin:18px 0;border-radius:12px}'
                'audio{width:290px;vertical-align:middle}td,th{padding:8px;border-bottom:1px solid #ddd}</style>'
                '<h1>Chatterbox S3Gen · baseline versus two flow-RL arms</h1>'
                + ('<p><strong>Work-in-progress snapshot:</strong> continue step 750; merged-new step 500. '
                   'Neither arm has completed the 4,000-step run.</p>' if args.interim else '')
                + f'<p>{len(cases)} UID-held-out emotion sources, {len(seeds)} matched seed(s), one seed shown below. '
                'The target audio supplies the voice identity and acoustic prompt; the source supplies words and emotion/style scores. '
                'AudioBox Production Quality (PQ) and top-three source-emotion MSE come from the same Ears model used '
                'for training, so they can be reward-hacked. Target CAMPPlus cosine is only a proxy for speaker similarity. '
                'Lower emotion MSE and higher other values are directionally better; listen before choosing a model.</p>'
                '<table><tr><th>Arm</th><th>Mean PQ ↑</th><th>Mean emotion MSE ↓</th><th>Mean target CAMP cosine ↑</th></tr>'
                + avg_rows + '</table>' + ''.join(cards) + '</html>')
    preview.write_text(document)
    print(json.dumps({'summary': summary, 'preview': str(preview)}), flush=True)


if __name__ == '__main__':
    main()
