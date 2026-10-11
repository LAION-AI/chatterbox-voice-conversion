#!/usr/bin/env python3
"""Matched-seed selection and listening evaluation for the three 24k VC arms.

Select one of steps 8k, 16k and 24k per arm on 40 UID-disjoint source clips,
then render all 160 clips with two fixed targets, the supervised baseline and
all three selected arms. Every GPU writes one packed TAR and one JSONL; no
per-sample files are created. Humaneness Ears is the *training* reward model,
so reported scores are proxies, not independent proof of listening quality.
"""
from __future__ import annotations

from collections import defaultdict
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
from scipy.signal import resample_poly
import soundfile as sf
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'grow_v1'))
from generate_listening import encode_mp3, load_model, ref_dict, set_adapter
from train_grow import EarsReward, conditioning_scores
from infer_followup import fold_linear_lora, install_trainables

OUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval')
V1 = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v1/training')
RUNS = {
    'continue': Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v2_ce/continue'),
    'remerged_new': Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v2_ce/remerged_new'),
    'speaker_mix': Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v3_ce_emo_speaker/speaker_mix'),
}
SCHEMAS = {'continue': 'chatterbox-s3gen-grow-v2-ddp',
           'remerged_new': 'chatterbox-s3gen-grow-v2-ddp',
           'speaker_mix': 'chatterbox-s3gen-grow-v3-ce-emo-speaker'}
CANDIDATES = (8000, 16000, 24000)
ARMS = ('baseline', 'continue', 'remerged_new', 'speaker_mix')


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def add_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    member.mtime = 0
    archive.addfile(member, io.BytesIO(content))


def read_wave(archive: tarfile.TarFile, member: str) -> np.ndarray:
    with archive.extractfile(member) as handle:
        wave, rate = sf.read(io.BytesIO(handle.read()), dtype='float32', always_2d=True)
    wave = wave.mean(axis=1)
    if rate != 24000:
        common = math.gcd(rate, 24000)
        wave = resample_poly(wave, 24000 // common, rate // common).astype(np.float32)
    return wave


def selection_cases(sources: list[dict]) -> list[str]:
    # One non-overlapping clip per emotion, then fill to 40 if necessary.
    manifest = load_json(OUT / 'site_inputs.json')
    by_case = {row['case_id']: row for row in sources}
    grouped = defaultdict(list)
    for case in manifest['cases']:
        if not by_case[case['case_id']]['rl_training_uid_overlap']:
            grouped[int(case['emotion_index'])].append(case)
    selected = [min(options, key=lambda case: (case['sample'], case['uid']))['case_id']
                for _, options in sorted(grouped.items()) if options]
    for case in manifest['cases']:
        cid = case['case_id']
        if (len(selected) < 40 and cid not in selected
                and not by_case[cid]['rl_training_uid_overlap']):
            selected.append(cid)
    assert len(selected) == 40 and len(set(selected)) == 40
    return selected


def load_arm(arm: str, device: torch.device):
    model = load_model(device)
    if arm == 'remerged_new':
        fold_linear_lora(model)
        parent = torch.load(V1 / 'merged_new/checkpoints/step-0004000.pt',
                            map_location='cpu', weights_only=True)
        assert parent['schema'] == 'chatterbox-s3gen-grow-v1'
        assert parent['step'] == 4000 and parent['arm'] == 'merged_new'
        install_trainables(model, parent['trainable'], device)
        fold_linear_lora(model)
        del parent
    return model.eval()


def use_checkpoint(model, arm: str, step: int, device: torch.device) -> None:
    if arm == 'baseline':
        assert step == 0
        return
    path = RUNS[arm] / 'checkpoints' / f'step-{step:07d}.pt'
    proof = load_json(path.with_suffix('.COMPLETE.json'))
    assert proof['status'] == 'COMPLETE' and proof['step'] == step
    assert path.stat().st_size == proof['bytes']
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['schema'] == SCHEMAS[arm] and state['arm'] == arm
    assert state['step'] == step and state['reward']['score'] == 'audiobox:CE'
    install_trainables(model, state['trainable'], device)
    del state


def source_rewards(scorer: EarsReward, source_rows: list[dict],
                   archive: tarfile.TarFile) -> dict[str, np.ndarray]:
    result = {}
    for i, row in enumerate(source_rows):
        wave = read_wave(archive, row['audio_member'])
        values = scorer.score(wave[None])[0]
        result[row['case_id']] = values
        if (i + 1) % 20 == 0:
            print(json.dumps({'event': 'SOURCE_SCORED', 'rank': dist.get_rank(),
                              'count': i + 1}), flush=True)
    return result


def synthesize(model, source: dict, target: dict, names: list[str],
               seed: int, device: torch.device) -> np.ndarray:
    scores = conditioning_scores(source, target, names)
    set_adapter(model, scores, 1.0, device)
    reference = ref_dict(target, device, acoustic_prompt=True)
    tokens = np.frombuffer(source['tokens'], dtype='<i2').astype(np.int64)
    token = torch.from_numpy(tokens[None]).to(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        mel = model.flow_inference(speech_tokens=token, ref_dict=dict(reference),
                                   n_cfm_timesteps=10, finalize=True)
        generated, _ = model.hift_inference(mel)
    wave = generated.float().cpu().numpy().reshape(-1)
    assert np.isfinite(wave).all() and len(wave) >= 24000
    return wave


def measurements(model, scorer: EarsReward, wave: np.ndarray,
                 source_ears: np.ndarray, target: dict,
                 device: torch.device) -> dict:
    output = scorer.score(wave[None])[0]
    top2 = np.argsort(source_ears[:40])[-2:]
    mse = float(np.mean((output[top2] - source_ears[top2]) ** 2))
    ce = float(output[scorer.norm['score_names'].index('audiobox:CE')])
    pq = float(output[scorer.pq_index])
    from score_inputs import to16
    with torch.inference_mode():
        camp = model.speaker_encoder.inference(
            torch.from_numpy(to16(wave, 24000)[None]).to(device)).float().cpu().numpy()[0]
    target_camp = np.frombuffer(target['camp'], dtype='<f2').astype(np.float32)
    camp /= max(float(np.linalg.norm(camp)), 1e-8)
    target_camp /= max(float(np.linalg.norm(target_camp)), 1e-8)
    return {'audiobox_ce': ce, 'audiobox_pq': pq,
            'top2_emotion_mse': mse, 'target_camp_cosine': float(camp @ target_camp),
            'top2_emotions': [scorer.names[int(i)] for i in top2[::-1]],
            'duration_s': len(wave) / 24000}


def evaluate(arm: str, step: int, cases: list[dict], targets: list[dict],
             names: list[str], source_scores: dict[str, np.ndarray],
             scorer: EarsReward, device: torch.device,
             archive: tarfile.TarFile | None, stage: str, rank: int, world: int) -> list[dict]:
    model = load_arm(arm, device)
    use_checkpoint(model, arm, step, device)
    results = []
    for index, case in enumerate(cases):
        if index % world != rank:
            continue
        for target in targets:
            seed = 20261011 + int(case['case_id'].split('-')[0]) * 10 + int(case['case_id'].split('-')[1])
            wave = synthesize(model, case, target, names, seed, device)
            record = {'stage': stage, 'arm': arm, 'step': step,
                      'case_id': case['case_id'], 'emotion': case['emotion'],
                      'source_uid': case['uid'], 'target_uid': target['uid'],
                      'target_label': target['target_label'], 'seed': seed,
                      'rl_training_uid_overlap': case['rl_training_uid_overlap'],
                      **measurements(model, scorer, wave, source_scores[case['case_id']],
                                     target, device)}
            if archive is not None:
                member = f'{case["case_id"]}/{arm}_vc_{target["target_label"]}.mp3'
                mp3, clipped = encode_mp3(wave)
                add_bytes(archive, member, mp3)
                record['mp3_member'] = member
                record['clipped_fraction'] = clipped
            results.append(record)
        if (index + 1) % 20 == 0:
            print(json.dumps({'stage': stage, 'arm': arm, 'step': step,
                              'rank': rank, 'through_case': index + 1}), flush=True)
    del model
    torch.cuda.empty_cache()
    return results


def write_jsonl(path: Path, rows: list[dict]) -> None:
    partial = path.with_suffix('.jsonl.partial')
    with partial.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    os.replace(partial, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def choose_checkpoints(rows: list[dict]) -> dict:
    baseline = [r for r in rows if r['arm'] == 'baseline']
    assert len(baseline) == 80
    keys = ('audiobox_ce', 'top2_emotion_mse', 'target_camp_cosine')
    spread = {key: max(float(np.std([r[key] for r in baseline])), .01)
              for key in keys}
    base_mean = {key: float(np.mean([r[key] for r in baseline])) for key in keys}
    selection = {'schema': 'grow-followup-40-case-heldout-selection-v1',
                 'criterion': '0.4*CE_gain/baseline_sd + 0.4*MSE_reduction/baseline_sd + 0.2*speaker_gain/baseline_sd',
                 'cases': 40, 'targets': 2, 'source_training_uid_overlap': 0,
                 'baseline_means': base_mean, 'baseline_sds': spread,
                 'arms': {}}
    for arm in RUNS:
        options = []
        for step in CANDIDATES:
            group = [r for r in rows if r['arm'] == arm and r['step'] == step]
            assert len(group) == 80, (arm, step, len(group))
            means = {key: float(np.mean([r[key] for r in group])) for key in keys}
            composite = (.4 * (means['audiobox_ce'] - base_mean['audiobox_ce']) / spread['audiobox_ce']
                         + .4 * (base_mean['top2_emotion_mse'] - means['top2_emotion_mse']) / spread['top2_emotion_mse']
                         + .2 * (means['target_camp_cosine'] - base_mean['target_camp_cosine']) / spread['target_camp_cosine'])
            options.append({'step': step, 'score': float(composite), 'means': means})
        winner = max(options, key=lambda option: (option['score'], option['step']))
        selection['arms'][arm] = {'selected_step': winner['step'],
                                  'candidates': options}
    return selection


def main() -> None:
    rank, world = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS'])
    assert world == 4 and 0 <= rank < 4 and torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.device_count() == 1
                          else 'cuda:' + os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', rank=rank, world_size=world)
    prep = load_json(OUT / 'prepared.COMPLETE.json')
    assert prep['status'] == 'COMPLETE' and prep['sources'] == 160 and prep['targets'] == 2
    for arm, path in RUNS.items():
        done = load_json(path / 'DONE.json')
        assert done['status'] == 'COMPLETE' and done['step'] == 24000, arm
    rows = pq.read_table(OUT / 'prepared.parquet').to_pylist()
    sources = sorted((row for row in rows if row['role'] == 'source'),
                     key=lambda row: row['case_id'])
    targets = sorted((row for row in rows if row['role'] == 'target'),
                     key=lambda row: row['target_label'])
    assert len(sources) == 160 and len(targets) == 2
    selected_ids = set(selection_cases(sources))
    selector = [row for row in sources if row['case_id'] in selected_ids]
    assert len(selector) == 40
    scorer = EarsReward(device)
    with tarfile.open(OUT / 'site_inputs.tar', 'r') as raw_archive:
        source_scores = source_rewards(scorer, [r for i, r in enumerate(sources)
                                                if i % world == rank], raw_archive)
    # Selection uses only disjoint sources; add any assigned selector cases
    # missed by the full-case rank partition if their ordering differs.
    missing = [row for i, row in enumerate(selector)
               if i % world == rank and row['case_id'] not in source_scores]
    if missing:
        with tarfile.open(OUT / 'site_inputs.tar', 'r') as raw_archive:
            source_scores.update(source_rewards(scorer, missing, raw_archive))
    chosen_rows = []
    chosen_rows += evaluate('baseline', 0, selector, targets, prep['score_order'],
                            source_scores, scorer, device, None, 'selection', rank, world)
    for arm in RUNS:
        for step in CANDIDATES:
            chosen_rows += evaluate(arm, step, selector, targets, prep['score_order'],
                                    source_scores, scorer, device, None,
                                    'selection', rank, world)
    write_jsonl(OUT / f'selection-rank{rank}.jsonl', chosen_rows)
    dist.barrier()
    selection_path = OUT / 'SELECTED.json'
    if rank == 0:
        combined = [row for index in range(world)
                    for row in read_jsonl(OUT / f'selection-rank{index}.jsonl')]
        assert len(combined) == 40 * 2 * (1 + 3 * len(CANDIDATES))
        selection = choose_checkpoints(combined)
        selection['selection_case_ids'] = sorted(selected_ids)
        temp = selection_path.with_suffix('.json.partial')
        temp.write_text(json.dumps(selection, indent=2) + '\n')
        os.replace(temp, selection_path)
        print(json.dumps({'event': 'SELECTED', 'arms': selection['arms']}), flush=True)
    dist.barrier()
    selection = load_json(selection_path)
    temp_tar = OUT / f'audio-rank{rank}.tar.partial'
    full_rows = []
    with tarfile.open(temp_tar, 'w') as audio_archive:
        for arm in ARMS:
            step = 0 if arm == 'baseline' else selection['arms'][arm]['selected_step']
            full_rows += evaluate(arm, step, sources, targets, prep['score_order'],
                                  source_scores, scorer, device, audio_archive,
                                  'full', rank, world)
    os.replace(temp_tar, OUT / f'audio-rank{rank}.tar')
    write_jsonl(OUT / f'full-rank{rank}.jsonl', full_rows)
    dist.barrier()
    if rank == 0:
        combined = [row for index in range(world)
                    for row in read_jsonl(OUT / f'full-rank{index}.jsonl')]
        assert len(combined) == 160 * 2 * len(ARMS)
        metrics = OUT / 'metrics.parquet'
        temporary = metrics.with_suffix('.parquet.partial')
        pq.write_table(pa.Table.from_pylist(combined), temporary, compression='zstd')
        os.replace(temporary, metrics)
        summary = {'status': 'COMPLETE', 'cases': 160, 'targets': 2,
                   'arms': list(ARMS), 'selected_steps': {arm: selection['arms'][arm]['selected_step']
                                                         for arm in RUNS},
                   'source_training_uid_overlap': prep['rl_training_source_uid_overlap'],
                   'metrics': {}}
        for arm in ARMS:
            part = [r for r in combined if r['arm'] == arm]
            nonoverlap = [r for r in part if not r['rl_training_uid_overlap']]
            summary['metrics'][arm] = {
                split: {key: float(np.mean([r[key] for r in subset]))
                        for key in ('audiobox_ce', 'audiobox_pq', 'top2_emotion_mse',
                                    'target_camp_cosine')}
                for split, subset in (('all', part), ('uid_disjoint', nonoverlap))}
        done = OUT / 'EVAL_DONE.json'
        temporary = done.with_suffix('.json.partial')
        temporary.write_text(json.dumps(summary, indent=2) + '\n')
        os.replace(temporary, done)
        print(json.dumps(summary), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
