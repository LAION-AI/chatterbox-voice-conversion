#!/usr/bin/env python3
"""Repack selected S3Gen training rows into one compact Parquet container."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(os.environ.get('CHATTERBOX_RUN_ROOT',
    '/e/scratch/reformo/schuhmann1_moss/out/chatterbox_s3gen_300h_v1'))
OUT = BASE / 'grow_v1'


def main() -> None:
    report = json.loads((OUT / 'SELECTION.json').read_text())
    assert report['status'] == 'SELECTED' and report['pairs'] == 4000
    pairs = pq.read_table(OUT / 'selection.parquet', columns=['source_uid', 'target_uid']).to_pylist()
    needed = {r['source_uid'] for r in pairs} | {r['target_uid'] for r in pairs}
    assert len(needed) >= report['unique_source_uids']
    result = []
    started = time.monotonic()
    for part in range(4):
        path = BASE / f'packed/train-{part}.parquet'
        reader = pq.ParquetFile(path)
        taken = 0
        for batch in reader.iter_batches(batch_size=512):
            for row in batch.to_pylist():
                if row['uid'] in needed:
                    result.append(row)
                    taken += 1
        print(json.dumps({'part': part, 'selected': taken,
                          'elapsed_s': time.monotonic() - started}), flush=True)
    assert len(result) == len(needed) == len({r['uid'] for r in result})
    by_uid = {r['uid']: r for r in result}
    for row in pairs:
        assert row['source_uid'] in by_uid and row['target_uid'] in by_uid
    target = OUT / 'pool.parquet'
    assert not target.exists(), 'Frozen pool already exists; do not overwrite silently'
    temporary = OUT / 'pool.parquet.partial'
    pq.write_table(pa.Table.from_pylist(result), temporary, compression='zstd', row_group_size=256)
    os.replace(temporary, target)
    proof = {'status': 'COMPLETE', 'rows': len(result), 'source_uids': report['unique_source_uids'],
             'target_uids': report['unique_target_uids'], 'bytes': target.stat().st_size,
             'elapsed_s': time.monotonic() - started, 'packed_files': 1,
             'loose_audio_files': 0}
    temporary = OUT / 'POOL_COMPLETE.json.partial'
    temporary.write_text(json.dumps(proof, indent=2) + '\n')
    os.replace(temporary, OUT / 'POOL_COMPLETE.json')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
