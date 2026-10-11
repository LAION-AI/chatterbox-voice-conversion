#!/usr/bin/env python3
"""Matched-seed evaluation of the completed 40k full-flow VC checkpoint.

Reuses the exact 160 source cases, two targets, scores, and seed formula of the
published 24k comparison. Four GPU ranks write four TARs and JSONLs, never
individual MP3 files. The curated 40 are for listening; the other 93
training-UID-disjoint cases support the main aggregate comparison.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'grow_publish'))
from eval_followups import (add_bytes, measurements, read_jsonl, source_rewards,
                            synthesize, write_jsonl)
from train_full import OUT, STEPS, load_merged_model
from generate_listening import encode_mp3
from train_grow import EarsReward

INPUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval')
EVAL = OUT / 'evaluation'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def load_full_model(device: torch.device, step: int):
    if step == STEPS:
        done = read_json(OUT / 'DONE.json')
        assert done['status'] == 'COMPLETE' and done['step'] == STEPS == 40000
    checkpoint = OUT / 'checkpoints' / f'step-{step:07d}.pt'
    proof = read_json(checkpoint.with_suffix('.COMPLETE.json'))
    assert proof['status'] == 'COMPLETE' and proof['step'] == step
    assert checkpoint.stat().st_size == proof['bytes']
    model, merge_error, n_params = load_merged_model(device)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['schema'] == 'chatterbox-s3gen-grow-v4-full-403030'
    assert state['step'] == step and state['group_size'] == 16
    trainable = {name: value for name, value in model.named_parameters()
                 if value.requires_grad}
    assert set(trainable) == set(state['trainable'])
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(state['trainable'][name].to(device))
    del state
    model.requires_grad_(False)
    model.eval()
    return model, merge_error, n_params


def averages(rows: list[dict]) -> dict:
    keys = ('audiobox_ce', 'audiobox_pq', 'top2_emotion_mse',
            'target_camp_cosine')
    assert rows
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-step', type=int, default=35000)
    args = parser.parse_args()
    global EVAL
    if args.smoke:
        EVAL = OUT / 'evaluation_smoke'
    step = args.smoke_step if args.smoke else STEPS
    rank = int(os.environ['SLURM_PROCID'])
    world = int(os.environ['SLURM_NTASKS'])
    assert world == 4 and 0 <= rank < world and torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.device_count() == 1
                          else 'cuda:' + os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', rank=rank, world_size=world)
    if rank == 0:
        EVAL.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    if not args.smoke and (EVAL / 'EVAL_DONE.json').exists():
        existing = read_json(EVAL / 'EVAL_DONE.json')
        assert existing['status'] == 'COMPLETE' and existing['step'] == STEPS
        dist.barrier()
        dist.destroy_process_group()
        return
    prep = read_json(INPUT / 'prepared.COMPLETE.json')
    assert prep['status'] == 'COMPLETE' and prep['sources'] == 160
    assert prep['targets'] == 2
    rows = pq.read_table(INPUT / 'prepared.parquet').to_pylist()
    sources = sorted((row for row in rows if row['role'] == 'source'),
                     key=lambda row: row['case_id'])
    targets = sorted((row for row in rows if row['role'] == 'target'),
                     key=lambda row: row['target_label'])
    assert len(sources) == 160 and len(targets) == 2
    selected = set(read_json(INPUT / 'space_stage/CURATED.json')['selected_case_ids'])
    assert len(selected) == 40
    if args.smoke:
        sources = sources[:4]
    scorer = EarsReward(device)
    my_sources = [row for index, row in enumerate(sources) if index % world == rank]
    with tarfile.open(INPUT / 'site_inputs.tar', 'r') as archive:
        source_scores = source_rewards(scorer, my_sources, archive)
    model, merge_error, n_params = load_full_model(device, step)
    output = EVAL / f'audio-rank{rank}.tar'
    temporary = output.with_suffix('.tar.partial')
    records = []
    with tarfile.open(temporary, 'w') as archive:
        for index, source in enumerate(sources):
            if index % world != rank:
                continue
            cid = source['case_id']
            for target in targets:
                label = target['target_label']
                seed = 20261011 + int(cid.split('-')[0]) * 10 + int(cid.split('-')[1])
                wave = synthesize(model, source, target, prep['score_order'],
                                  seed, device)
                measured = measurements(model, scorer, wave,
                                        source_scores[cid], target, device)
                member = f'{cid}/full40k_vc_{label}.mp3'
                mp3, clipped = encode_mp3(wave)
                add_bytes(archive, member, mp3)
                records.append({'arm': 'full40k', 'step': step,
                                'case_id': cid, 'emotion': source['emotion'],
                                'source_uid': source['uid'],
                                'target_uid': target['uid'],
                                'target_label': label, 'seed': seed,
                                'rl_training_uid_overlap': bool(source['rl_training_uid_overlap']),
                                'curated': cid in selected,
                                'mp3_member': member, 'clipped_fraction': clipped,
                                **measured})
            if len(records) % 40 == 0:
                print(json.dumps({'event': 'EVAL_PROGRESS', 'rank': rank,
                                  'predictions': len(records)}), flush=True)
    assert len(records) == (2 if args.smoke else 80)
    os.replace(temporary, output)
    write_jsonl(EVAL / f'metrics-rank{rank}.jsonl', records)
    print(json.dumps({'event': 'RANK_COMPLETE', 'rank': rank,
                      'predictions': len(records), 'merge_error': merge_error,
                      'flow_parameters': n_params}), flush=True)
    dist.barrier()
    if rank == 0:
        combined = [row for worker in range(world)
                    for row in read_jsonl(EVAL / f'metrics-rank{worker}.jsonl')]
        expected = 8 if args.smoke else 160 * 2
        assert len(combined) == expected
        assert len({(r['case_id'], r['target_label']) for r in combined}) == expected
        if args.smoke:
            atomic_json(EVAL / 'EVAL_SMOKE.json',
                        {'status': 'PASS', 'checkpoint_step': step,
                         'predictions': expected, 'metrics': averages(combined),
                         'packed_tars': 4})
            print(json.dumps({'event': 'EVAL_SMOKE_PASS',
                              'checkpoint_step': step,
                              'predictions': expected}), flush=True)
            dist.barrier()
            dist.destroy_process_group()
            return
        metrics = EVAL / 'metrics.parquet'
        pq.write_table(pa.Table.from_pylist(combined),
                       str(metrics) + '.partial', compression='zstd')
        os.replace(str(metrics) + '.partial', metrics)
        from finalize_eval import finalize
        summary = finalize()
        print(json.dumps({'event': 'EVAL_DONE', 'step': STEPS,
                          'predictions': 320,
                          'uid_disjoint_noncurated_cases': summary['uid_disjoint_noncurated_case_count']}),
              flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
