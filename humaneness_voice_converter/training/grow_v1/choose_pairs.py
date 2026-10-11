#!/usr/bin/env python3
"""Choose 100 strong training-source prompts per emotion and diverse target voices.

This only reads small Parquet metadata columns. No audio is decoded here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(os.environ.get('CHATTERBOX_RUN_ROOT',
    '/e/scratch/reformo/schuhmann1_moss/out/chatterbox_s3gen_300h_v1'))
OUT = BASE / 'grow_v1'
SELECTED_PER_EMOTION = 100
TARGET_POOL = 500
SEED = 20261009


def hash_order(uid: str) -> bytes:
    return hashlib.sha256(f'grow-v1|{uid}'.encode()).digest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proof = json.loads((BASE / 'packed/train-0.COMPLETE.json').read_text())
    names = proof['score_order'][:40]
    assert len(names) == 40 and len(set(names)) == 40
    sources = []
    target_pool = []
    paths = [BASE / f'packed/train-{part}.parquet' for part in range(4)]
    for part, path in enumerate(paths):
        table = pq.read_table(path, columns=['uid', 'dur_s', 'scores', 'camp', 'lang'])
        for record in table.to_pylist():
            uid, duration = record['uid'], float(record['dur_s'])
            values = np.frombuffer(record['scores'], dtype='<f4')
            camp = np.frombuffer(record['camp'], dtype='<f2').astype(np.float32)
            assert values.shape == (99,) and camp.shape == (192,)
            assert np.isfinite(values).all() and np.isfinite(camp).all()
            unit = camp / max(1e-8, np.linalg.norm(camp))
            item = {'uid': uid, 'part': part, 'dur_s': duration, 'lang': record['lang'],
                    'emotion_scores': values[:40].copy(), 'camp_unit': unit}
            if 3 <= duration <= 8:
                sources.append(item)
            if 5 <= duration <= 10:
                target_pool.append(item)
    assert len(sources) > 10000 and len(target_pool) > 10000
    target_pool.sort(key=lambda item: hash_order(item['uid']))
    # A large-enough fixed pool avoids reusing just a few target voices.
    targets = target_pool[:TARGET_POOL]
    assert len({item['uid'] for item in targets}) == TARGET_POOL
    rng = np.random.default_rng(SEED)
    pairs, floor = [], {}
    for axis, name in enumerate(names):
        ranked = sorted(sources, key=lambda item: (-float(item['emotion_scores'][axis]),
                                                    hash_order(item['uid'])))
        selected = ranked[:SELECTED_PER_EMOTION]
        assert len(selected) == SELECTED_PER_EMOTION
        floor[name] = float(selected[-1]['emotion_scores'][axis] * 6)
        for source in selected:
            start = int(rng.integers(0, TARGET_POOL))
            target = next((targets[(start + k) % TARGET_POOL] for k in range(TARGET_POOL)
                           if targets[(start + k) % TARGET_POOL]['uid'] != source['uid']
                           and float(np.dot(targets[(start + k) % TARGET_POOL]['camp_unit'],
                                            source['camp_unit'])) < .30), None)
            assert target is not None, source['uid']
            pairs.append({'axis': axis, 'emotion': name, 'source_uid': source['uid'],
                          'target_uid': target['uid'], 'source_part': source['part'],
                          'target_part': target['part'], 'source_lang': source['lang'],
                          'source_annotated_axis_raw': float(source['emotion_scores'][axis] * 6),
                          'source_target_camp_cosine': float(np.dot(source['camp_unit'], target['camp_unit']))})
    assert len(pairs) == 40 * SELECTED_PER_EMOTION
    assert len({(r['axis'], r['source_uid']) for r in pairs}) == len(pairs)
    table = pa.Table.from_pylist(pairs)
    temporary = OUT / 'selection.parquet.partial'
    pq.write_table(table, temporary, compression='zstd')
    temporary.replace(OUT / 'selection.parquet')
    report = {'schema': 'chatterbox-s3gen-grow-selection-v1', 'status': 'SELECTED',
              'pairs': len(pairs), 'source_eligible': len(sources), 'target_eligible': len(target_pool),
              'unique_source_uids': len({r['source_uid'] for r in pairs}),
              'unique_target_uids': len({r['target_uid'] for r in pairs}),
              'source_duration_s': [3, 8], 'target_duration_s': [5, 10],
              'source_target_camp_cosine_max': max(r['source_target_camp_cosine'] for r in pairs),
              'emotion_axis_order': names, 'top100_min_annotated_score_raw': floor,
              'seed': SEED, 'input_files': [str(path) for path in paths]}
    temporary = OUT / 'SELECTION.json.partial'
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(OUT / 'SELECTION.json')
    print(json.dumps({k: v for k, v in report.items() if k not in ('emotion_axis_order',
                         'top100_min_annotated_score_raw', 'input_files')}), flush=True)


if __name__ == '__main__':
    main()
