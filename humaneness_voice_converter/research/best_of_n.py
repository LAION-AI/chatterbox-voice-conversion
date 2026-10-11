#!/usr/bin/env python3
"""Nested best-of-1/4/8/16 on the published 40-case curated VC set.

Four GPU ranks generate 16 seeds for each assigned source/target pair. The
selection reward gives one third each to standardized Audiobox Content
Enjoyment, target CAMPPlus cosine, and negative top-two source-emotion MSE.
Standardization is fixed from the earlier UID-disjoint, noncurated full40k
evaluation; the candidates being ranked cannot set their own scale.

Only winning MP3s are stored, in rank-wise TARs. All 1,280 candidate metrics,
seeds, rewards and measured latencies are retained in Parquet for audit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'grow_v4'))
sys.path.insert(0, str(PROJECT / 'grow_publish'))
from eval_after_full import load_full_model
from eval_followups import add_bytes, measurements, source_rewards, synthesize
from generate_listening import encode_mp3
from train_grow import EarsReward

INPUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval')
EVAL = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v4_full_403030/evaluation')
OUT = Path('/e/scratch/laionize/schuhmann1/hvc_best_of_n_2026_10_11')
NS = (1, 4, 8, 16)
FIELDS = ('audiobox_ce', 'target_camp_cosine', 'top2_emotion_mse')


def calibrate() -> dict:
    rows = pq.read_table(EVAL / 'metrics.parquet').to_pylist()
    heldout = [row for row in rows if row['arm'] == 'full40k'
               and not row['curated'] and not row['rl_training_uid_overlap']]
    assert len(heldout) >= 100, len(heldout)
    summary = {}
    for key in FIELDS:
        values = np.asarray([float(row[key]) for row in heldout], np.float64)
        summary[key] = {'mean': float(np.mean(values)),
                        'std': float(max(np.std(values), .01))}
    return {'source': str(EVAL / 'metrics.parquet'), 'heldout_predictions': len(heldout),
            'formula': '(z(CE) + z(target CAMPPlus cosine) - z(top-two source-emotion MSE))/3',
            'terms': summary}


def reward(row: dict, calibration: dict) -> float:
    terms = calibration['terms']
    def z(key: str) -> float:
        return ((float(row[key]) - terms[key]['mean']) / terms[key]['std'])
    return float((z('audiobox_ce') + z('target_camp_cosine')
                  - z('top2_emotion_mse')) / 3.0)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, path)


def write_parquet(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression='zstd')
    os.replace(temporary, path)


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> dict:
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='One source per GPU, two targets')
    args = parser.parse_args()
    rank = int(os.environ['SLURM_PROCID'])
    world = int(os.environ['SLURM_NTASKS'])
    assert world == 4 and 0 <= rank < world and torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.device_count() == 1
                          else 'cuda:' + os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', rank=rank, world_size=world)
    stage = OUT / ('smoke' if args.smoke else 'full')
    if rank == 0:
        stage.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    calibration = calibrate()
    prep = json.loads((INPUT / 'prepared.COMPLETE.json').read_text())
    assert prep['status'] == 'COMPLETE' and prep['sources'] == 160
    selected = set(json.loads((INPUT / 'space_stage/CURATED.json').read_text())['selected_case_ids'])
    assert len(selected) == 40
    prepared = pq.read_table(INPUT / 'prepared.parquet').to_pylist()
    sources = sorted((row for row in prepared
                      if row['role'] == 'source' and row['case_id'] in selected),
                     key=lambda row: row['case_id'])
    targets = sorted((row for row in prepared if row['role'] == 'target'),
                     key=lambda row: row['target_label'])
    assert len(sources) == 40 and len(targets) == 2
    assigned = [row for index, row in enumerate(sources) if index % world == rank]
    if args.smoke:
        assigned = assigned[:1]
    scorer = EarsReward(device)
    with tarfile.open(INPUT / 'site_inputs.tar', 'r') as input_tar:
        source_scores = source_rewards(scorer, assigned, input_tar)
    model, merge_error, flow_params = load_full_model(device, 40000)
    archive_path = stage / f'winners-rank{rank}.tar'
    temp_tar = archive_path.with_suffix('.tar.partial')
    candidates: list[dict] = []
    winners: list[dict] = []
    run_start = time.perf_counter()
    with tarfile.open(temp_tar, 'w') as archive:
        for source in assigned:
            cid = source['case_id']
            for target in targets:
                label = target['target_label']
                base_seed = (20261011 + int(cid.split('-')[0]) * 10
                             + int(cid.split('-')[1]))
                waves = []
                group = []
                for index in range(16):
                    seed = base_seed + 100003 * index
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    wave = synthesize(model, source, target, prep['score_order'],
                                      seed, device)
                    torch.cuda.synchronize(device)
                    t1 = time.perf_counter()
                    measured = measurements(model, scorer, wave,
                                            source_scores[cid], target, device)
                    torch.cuda.synchronize(device)
                    t2 = time.perf_counter()
                    record = {'case_id': cid, 'source_uid': source['uid'],
                              'emotion': source['emotion'], 'target_label': label,
                              'target_uid': target['uid'], 'candidate_index': index,
                              'seed': seed, 'rl_training_uid_overlap': bool(source['rl_training_uid_overlap']),
                              'inference_seconds': t1 - t0,
                              'scoring_seconds': t2 - t1, **measured}
                    record['selection_reward'] = reward(record, calibration)
                    group.append(record)
                    candidates.append(record)
                    waves.append(wave)
                for n in NS:
                    # Nested pools; deterministic lowest-index tie break.
                    winner = max(group[:n], key=lambda item: (item['selection_reward'],
                                                                -item['candidate_index']))
                    member = f'{cid}/bon{n}_vc_{label}.mp3'
                    mp3, clipped = encode_mp3(waves[winner['candidate_index']])
                    add_bytes(archive, member, mp3)
                    winners.append({**winner, 'pool_size': n, 'mp3_member': member,
                                    'clipped_fraction': clipped,
                                    'pool_inference_seconds': float(sum(r['inference_seconds'] for r in group[:n])),
                                    'pool_scoring_seconds': float(sum(r['scoring_seconds'] for r in group[:n]))})
                print(json.dumps({'event': 'BON_PAIR_COMPLETE', 'rank': rank,
                                  'case_id': cid, 'target': label,
                                  'candidate_count': len(group),
                                  'best16_reward': winners[-1]['selection_reward']}), flush=True)
    os.replace(temp_tar, archive_path)
    write_parquet(stage / f'candidates-rank{rank}.parquet', candidates)
    write_parquet(stage / f'winners-rank{rank}.parquet', winners)
    atomic_json(stage / f'proof-rank{rank}.json',
                {'status': 'COMPLETE', 'rank': rank, 'cases': len(assigned),
                 'targets': len(targets), 'candidates': len(candidates),
                 'winners': len(winners), 'wall_seconds': time.perf_counter() - run_start,
                 'gpu': torch.cuda.get_device_name(device),
                 'merge_max_abs_error': merge_error, 'flow_parameters': flow_params})
    dist.barrier()
    if rank == 0:
        all_candidates = [row for worker in range(world) for row in
                          pq.read_table(stage / f'candidates-rank{worker}.parquet').to_pylist()]
        all_winners = [row for worker in range(world) for row in
                       pq.read_table(stage / f'winners-rank{worker}.parquet').to_pylist()]
        expected_pairs = 8 if args.smoke else 80
        assert len(all_candidates) == expected_pairs * 16
        assert len(all_winners) == expected_pairs * len(NS)
        write_parquet(stage / 'all_candidates.parquet', all_candidates)
        write_parquet(stage / 'all_winners.parquet', all_winners)
        metrics = ('selection_reward', 'audiobox_ce', 'audiobox_pq',
                   'top2_emotion_mse', 'target_camp_cosine',
                   'pool_inference_seconds', 'pool_scoring_seconds')
        summary = {str(n): aggregate([row for row in all_winners if row['pool_size'] == n],
                                     metrics) for n in NS}
        assert all(summary[str(b)]['selection_reward'] >=
                   summary[str(a)]['selection_reward'] - 1e-8
                   for a, b in zip(NS, NS[1:]))
        proof = {'status': 'COMPLETE', 'smoke': args.smoke,
                 'checkpoint_step': 40000, 'source_cases': expected_pairs // 2,
                 'target_voices': 2, 'candidate_pool_max': 16,
                 'candidate_predictions': len(all_candidates),
                 'winners': len(all_winners), 'packed_tars': 4,
                 'reward_calibration': calibration, 'summary': summary,
                 'caveat': 'The ranking reward reuses Humaneness Ears Medium, which was also used for RL training; this is not an independent or human listening test. Candidate latency is measured sequentially, not batched.'}
        atomic_json(stage / 'DONE.json', proof)
        print(json.dumps({'event': 'BON_DONE', 'pairs': expected_pairs,
                          'candidates': len(all_candidates),
                          'summary': summary}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
