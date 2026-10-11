#!/usr/bin/env python3
"""Validate packed 40k VC predictions and finish the matched comparison.

This CPU-only finalizer is safe to rerun. It deliberately derives holdout
counts from source metadata; no curation or overlap count is hard-coded.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tarfile

import numpy as np
import pyarrow.parquet as pq

OUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v4_full_403030')
EVAL = OUT / 'evaluation'
OLD = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval')
SCORE_KEYS = ('audiobox_ce', 'audiobox_pq', 'top2_emotion_mse',
              'target_camp_cosine')


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def mean_scores(rows: list[dict]) -> dict:
    assert rows
    result = {key: float(np.mean([float(row[key]) for row in rows]))
              for key in SCORE_KEYS}
    assert all(np.isfinite(value) for value in result.values())
    return result


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def finalize() -> dict:
    proof = read_json(OUT / 'DONE.json')
    assert proof['status'] == 'COMPLETE' and proof['step'] == 40000
    checkpoint = OUT / 'checkpoints/step-0040000.pt'
    ckproof = read_json(checkpoint.with_suffix('.COMPLETE.json'))
    assert ckproof['status'] == 'COMPLETE' and ckproof['step'] == 40000
    assert checkpoint.stat().st_size == ckproof['bytes']
    selected = set(read_json(OLD / 'space_stage/CURATED.json')['selected_case_ids'])
    assert len(selected) == 40
    source_rows = [r for r in pq.read_table(OLD / 'prepared.parquet').to_pylist()
                   if r['role'] == 'source']
    expected_cases = {r['case_id'] for r in source_rows}
    assert len(expected_cases) == 160 and selected <= expected_cases
    records = pq.read_table(EVAL / 'metrics.parquet').to_pylist()
    assert len(records) == 320
    keys = {(r['case_id'], r['target_label']) for r in records}
    assert keys == {(case, target) for case in expected_cases
                    for target in ('A', 'B')}
    for row in records:
        assert row['arm'] == 'full40k' and row['step'] == 40000
        assert row['curated'] == (row['case_id'] in selected)
        emotion_index, sample = map(int, row['case_id'].split('-'))
        assert row['seed'] == 20261011 + emotion_index * 10 + sample
        assert all(np.isfinite(float(row[key])) for key in SCORE_KEYS)
    members = set()
    for rank in range(4):
        with tarfile.open(EVAL / f'audio-rank{rank}.tar', 'r') as archive:
            names = archive.getnames()
            assert len(names) == 80 and len(set(names)) == 80
            members.update(names)
        lines = (EVAL / f'metrics-rank{rank}.jsonl').read_text().splitlines()
        assert len(lines) == 80
        rank_keys = {(r['case_id'], r['target_label']) for r in map(json.loads, lines)}
        assert len(rank_keys) == 80 and rank_keys <= keys
    assert len(members) == 320
    assert members == {r['mp3_member'] for r in records}

    holdout = [r for r in records
               if not r['rl_training_uid_overlap'] and not r['curated']]
    holdout_cases = {r['case_id'] for r in holdout}
    assert len(holdout_cases) == 93 and len(holdout) == 186
    all_uid_disjoint = [r for r in records if not r['rl_training_uid_overlap']]
    assert len({r['case_id'] for r in all_uid_disjoint}) == 123
    old = pq.read_table(OLD / 'metrics.parquet').to_pylist()
    matched_old = {}
    for arm in ('baseline', 'speaker_mix'):
        subset = [r for r in old if r['arm'] == arm and r['stage'] == 'full'
                  and r['case_id'] in holdout_cases]
        assert {(r['case_id'], r['target_label']) for r in subset} == \
            {(r['case_id'], r['target_label']) for r in holdout}
        matched_old[arm] = mean_scores(subset)
    summary = {
        'status': 'COMPLETE', 'step': 40000, 'sources': 160, 'targets': 2,
        'predictions': 320, 'checkpoint': str(checkpoint),
        'same_cases_targets_seeds_as_24k': True,
        'seed_formula': '20261011 + emotion_index*10 + sample',
        'curated_case_count': 40,
        'uid_disjoint_noncurated_case_count': len(holdout_cases),
        'uid_disjoint_all_case_count': 123,
        'metrics': {
            'all': mean_scores(records),
            'curated': mean_scores([r for r in records if r['curated']]),
            'uid_disjoint_noncurated': mean_scores(holdout),
            'uid_disjoint_all': mean_scores(all_uid_disjoint),
        },
        'matched_24k_on_uid_disjoint_noncurated': matched_old,
        'limitations': [
            'Humaneness Ears Medium CE, PQ and emotion scores are reward-model proxies, not independent human ratings.',
            'CAMPPlus cosine is a speaker-similarity proxy, not a verified speaker-ID result.',
            'UID-disjoint does not guarantee speaker-disjoint.',
            'The 93-case comparison excludes the 40 quality-curated listening cases; all three model arms use exactly the same 93 cases and two targets.',
        ],
    }
    write_json(EVAL / 'EVAL_DONE.json', summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(finalize(), indent=2, ensure_ascii=False))
