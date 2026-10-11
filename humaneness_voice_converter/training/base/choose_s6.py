"""Choose ~300 unique S6 hours including sidecar; emit four packed worklists."""
import collections
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SC = Path(os.environ.get('CHATTERBOX_SCRATCH', '/e/scratch/reformo/schuhmann1_moss'))
ROOT = Path(os.environ.get('CHATTERBOX_RUN_ROOT', str(SC / 'out/chatterbox_s3gen_300h_v1')))
SOURCE = Path(os.environ.get('CHATTERBOX_S6_MANIFEST', str(
    SC / 'out/m2_600m_ladder/manifests/S6/manifest.json')))


def h(uid):
    return hashlib.sha256(uid.encode()).digest()


def main():
    spec = json.loads(SOURCE.read_text())
    assert spec['status'] == 'complete' and spec['ladder_stage'] == 'tier2'
    assert spec['sidecar_stage'] == 'tier2'
    records = SOURCE.parent / spec['records']
    rows = {}
    with records.open() as stream:
        for line in stream:
            row = json.loads(line)
            rows.setdefault(row['uid'], row)
    assert len(rows) > 200000
    sources = collections.defaultdict(list)
    for row in rows.values():
        if not 2.0 <= float(row['dur_s']) <= 25.0:
            continue
        if not row.get('record_locator') or not row.get('target_locator'):
            continue
        sources[row['source']].append(row)
    assert 'sidecar' in sources
    # SHA order makes the split independent of manifest presentation ordering.
    for items in sources.values():
        items.sort(key=lambda r: h(r['uid']))
    total = sum(sum(r['dur_s'] for r in v) for v in sources.values())
    assert total > 300 * 3600
    # Deliberately oversample the relatively small synthetic sidecar to 30 h.
    sidecar_target = 30 * 3600
    other_total = sum(sum(r['dur_s'] for r in v) for k, v in sources.items() if k != 'sidecar')
    train = []
    remainder = []
    for source, items in sorted(sources.items()):
        available = sum(r['dur_s'] for r in items)
        target = sidecar_target if source == 'sidecar' else (300 * 3600 - sidecar_target) * available / other_total
        seconds = 0.0
        for row in items:
            if seconds < target:
                train.append(row)
                seconds += row['dur_s']
            else:
                remainder.append(row)
    train_seconds = sum(r['dur_s'] for r in train)
    assert 300 * 3600 <= train_seconds < 300 * 3600 + len(sources) * 25
    assert sum(r['dur_s'] for r in train if r['source'] == 'sidecar') / 3600 >= 30
    # UID-disjoint holdout, not a claim of speaker-disjoint generalization.
    remainder.sort(key=lambda r: h('heldout:' + r['uid']))
    heldout = []
    heldout_seconds = 0.0
    for row in remainder:
        if heldout_seconds >= 2 * 3600:
            break
        heldout.append(row)
        heldout_seconds += row['dur_s']
    assert heldout_seconds >= 2 * 3600
    # TAR locality and greedy duration balance across the four GPUs.
    bins = [([], 0.0) for _ in range(4)]
    for row in sorted(train, key=lambda r: (r['record_locator']['path'], h(r['uid']))):
        idx = min(range(4), key=lambda i: bins[i][1])
        bins[idx][0].append(row)
        bins[idx] = (bins[idx][0], bins[idx][1] + row['dur_s'])
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / 'SELECTION.json').exists(), 'Selection already frozen; refuse overwrite'
    for rank, (items, _) in enumerate(bins):
        pq.write_table(pa.Table.from_pylist(items), ROOT / f'train-{rank}.parquet', compression='zstd')
    pq.write_table(pa.Table.from_pylist(heldout), ROOT / 'heldout.parquet', compression='zstd')
    summary = {
        'status': 'SELECTED', 'source_manifest': str(SOURCE),
        'source_manifest_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'unique_source_rows': len(rows), 'unique_source_hours': sum(r['dur_s'] for r in rows.values()) / 3600,
        'train_rows': len(train), 'train_hours': train_seconds / 3600,
        'heldout_rows': len(heldout), 'heldout_hours': heldout_seconds / 3600,
        'train_by_source_hours': {k: sum(r['dur_s'] for r in train if r['source'] == k) / 3600 for k in sorted(sources)},
        'rank_hours': [sec / 3600 for _, sec in bins],
        'speaker_disjoint_holdout': False,
        'new_per_sample_files': 0,
    }
    (ROOT / 'SELECTION.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
