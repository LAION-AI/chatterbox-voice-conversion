#!/usr/bin/env python3
"""One-case, two-target inference smoke before the 160-case evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import tarfile

import pyarrow.parquet as pq
import torch

from eval_followups import (OUT, EarsReward, load_arm, measurements, read_wave,
                            synthesize, use_checkpoint)


def main() -> None:
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    proof = json.loads((OUT / 'prepared.COMPLETE.json').read_text())
    assert proof['status'] == 'COMPLETE' and proof['rows'] == 162
    rows = pq.read_table(OUT / 'prepared.parquet').to_pylist()
    source = next(row for row in rows if row['role'] == 'source'
                  and not row['rl_training_uid_overlap'])
    targets = sorted((row for row in rows if row['role'] == 'target'),
                     key=lambda row: row['target_label'])
    scorer = EarsReward(device)
    with tarfile.open(OUT / 'site_inputs.tar', 'r') as archive:
        wave = read_wave(archive, source['audio_member'])
    source_ears = scorer.score(wave[None])[0]
    result = {'status': 'PASS', 'case_id': source['case_id'], 'arms': {}}
    for arm in ('baseline', 'continue', 'remerged_new', 'speaker_mix'):
        model = load_arm(arm, device)
        step = 0 if arm == 'baseline' else 8000
        use_checkpoint(model, arm, step, device)
        result['arms'][arm] = []
        for target in targets:
            output = synthesize(model, source, target, proof['score_order'], 20261011, device)
            metric = measurements(model, scorer, output, source_ears, target, device)
            assert all(map(lambda key: abs(metric[key]) < 1e5,
                           ('audiobox_ce', 'audiobox_pq', 'top2_emotion_mse',
                            'target_camp_cosine')))
            result['arms'][arm].append({'target': target['target_label'], **metric})
        del model
        torch.cuda.empty_cache()
    destination = OUT / 'EVAL_SMOKE.json'
    temporary = destination.with_suffix('.json.partial')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(destination)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
