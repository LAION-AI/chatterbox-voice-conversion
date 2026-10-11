#!/usr/bin/env python3
"""Matched 40-case, two-target benchmark for independently runnable VC models.

Each of four GPU ranks loads one external converter once, reuses two target
files and one rotating source file, scores each output with the *same* frozen
Ears Medium and CAMPPlus implementations as the Humaneness 40k evaluation,
and writes one MP3 TAR and one Parquet shard. No per-sample output files.
This is only a proxy benchmark, not a subjective listening or WER study.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import tarfile
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import resample_poly
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / 'grow_v4'))
sys.path.insert(0, str(PROJECT / 'grow_publish'))
from eval_after_full import load_full_model
from eval_followups import add_bytes, measurements, source_rewards
from generate_listening import encode_mp3
from train_grow import EarsReward

BENCH = Path('/e/scratch/laionize/schuhmann1/hvc_benchmark_2026_10_11')
INPUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval')
ARMS = ('seed_v1', 'seed_v2', 'xvc')


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, path)


def write_parquet(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    pq.write_table(pa.Table.from_pylist(records), temporary, compression='zstd')
    os.replace(temporary, path)


def to24(wave: np.ndarray, rate: int) -> np.ndarray:
    wave = np.asarray(wave, np.float32).squeeze()
    assert wave.ndim == 1 and 1 <= len(wave) / rate <= 30
    assert np.isfinite(wave).all()
    if rate != 24000:
        g = math.gcd(rate, 24000)
        wave = resample_poly(wave, 24000 // g, rate // g).astype(np.float32)
    return wave


def seed_v1_loader():
    repo = BENCH / 'seed-vc'
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import inference as seed
    args = argparse.Namespace(source='', target='', output='', diffusion_steps=25,
                              length_adjust=1.0, inference_cfg_rate=0.7,
                              f0_condition=False, auto_f0_adjust=False,
                              semi_tone_shift=0, checkpoint=None, config=None,
                              fp16=True, collect_wave=True)
    loaded = seed.load_models(args)
    seed.load_models = lambda _: loaded

    def convert(source: Path, target: Path) -> tuple[np.ndarray, int]:
        args.source, args.target = str(source), str(target)
        return seed.main(args)

    return convert, {'variant': 'offline VC v1, Whisper-small/BigVGAN',
                     'diffusion_steps': 25, 'cfg_rate': 0.7,
                     'source': 'https://github.com/Plachtaa/seed-vc'}


def seed_v2_loader():
    repo = BENCH / 'seed-vc'
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import inference_v2 as seed
    weights = (BENCH / 'hf_cache/hub/models--Plachta--Seed-VC/snapshots/'
               '257283f9f41585055e8f858fba4fd044e5caed6e/v2')
    assert (weights / 'ar_base.pth').is_file() and (weights / 'cfm_small.pth').is_file()
    args = argparse.Namespace(diffusion_steps=25, length_adjust=1.0,
                              intelligibility_cfg_rate=0.7,
                              similarity_cfg_rate=0.7, top_p=0.9,
                              temperature=1.0, repetition_penalty=1.0,
                              convert_style=False, anonymization_only=False,
                              ar_checkpoint_path=str(weights / 'ar_base.pth'),
                              cfm_checkpoint_path=str(weights / 'cfm_small.pth'),
                              compile=False)
    seed.vc_wrapper_v2 = seed.load_v2_models(args)

    def convert(source: Path, target: Path) -> tuple[np.ndarray, int]:
        rate, wave = seed.convert_voice_v2(str(source), str(target), args)
        return wave, rate

    return convert, {'variant': 'VC v2, ASTRAL/CFM, timbre-only mode',
                     'diffusion_steps': 25, 'intelligibility_cfg_rate': 0.7,
                     'similarity_cfg_rate': 0.7, 'convert_style': False,
                     'source': 'https://github.com/Plachtaa/seed-vc'}


def xvc_loader():
    repo = BENCH / 'X-VC'
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from bins.infer_utils import (load_pair_as_tensors, load_xvc,
                                  run_offline, to_numpy_audio)
    cfg, model, device = load_xvc('configs/xvc.yaml', 'ckpts/xvc.pt', 0, False)
    rate = int(cfg['sample_rate'])

    def convert(source: Path, target: Path) -> tuple[np.ndarray, int]:
        source_wav, target_wav, target_cond = load_pair_as_tensors(
            str(source), str(target), cfg, device, 1280, False)
        with torch.inference_mode():
            output = run_offline(model, source_wav, target_wav, target_cond)
        return to_numpy_audio(output), rate

    return convert, {'variant': 'X-VC official offline checkpoint',
                     'source': 'https://github.com/Jerrister/X-VC',
                     'checkpoint': 'chenxie95/X-VC'}


LOADERS = {'seed_v1': seed_v1_loader, 'seed_v2': seed_v2_loader,
           'xvc': xvc_loader}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    rank, world = int(os.environ['SLURM_PROCID']), int(os.environ['SLURM_NTASKS'])
    assert world == 4 and 0 <= rank < world and torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.device_count() == 1
                          else 'cuda:' + os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', rank=rank, world_size=world)
    stage = BENCH / 'evaluation' / args.arm / ('smoke' if args.smoke else 'full')
    if rank == 0:
        stage.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    selected = set(json.loads((INPUT / 'space_stage/CURATED.json').read_text())['selected_case_ids'])
    assert len(selected) == 40
    prepared = pq.read_table(INPUT / 'prepared.parquet').to_pylist()
    sources = sorted((row for row in prepared
                      if row['role'] == 'source' and row['case_id'] in selected),
                     key=lambda row: row['case_id'])
    targets = sorted((row for row in prepared if row['role'] == 'target'),
                     key=lambda row: row['target_label'])
    assigned = [row for index, row in enumerate(sources) if index % world == rank]
    if args.smoke:
        assigned = assigned[:1]
    assert len(targets) == 2 and len(assigned) == (1 if args.smoke else 10)
    scorer = EarsReward(device)
    with tarfile.open(INPUT / 'site_inputs.tar', 'r') as raw:
        source_scores = source_rewards(scorer, assigned, raw)
    # The same frozen Chatterbox CAMPPlus scores *all* external outputs.
    scorer_model, _, _ = load_full_model(device, 40000)
    converter, metadata = LOADERS[args.arm]()
    records = []
    archive_path = stage / f'audio-rank{rank}.tar'
    partial_tar = archive_path.with_suffix('.tar.partial')
    with tempfile.TemporaryDirectory(prefix=f'{args.arm}-rank{rank}-', dir=stage) as tmp:
        temp = Path(tmp)
        source_file = temp / 'source.mp3'
        target_files = {label: temp / f'target-{label}.mp3' for label in ('A', 'B')}
        with tarfile.open(INPUT / 'site_inputs.tar', 'r') as raw:
            for label, path in target_files.items():
                with raw.extractfile(f'references/{label}.mp3') as handle:
                    path.write_bytes(handle.read())
            with tarfile.open(partial_tar, 'w') as output:
                for source in assigned:
                    cid = source['case_id']
                    with raw.extractfile(source['audio_member']) as handle:
                        source_file.write_bytes(handle.read())
                    for target in targets:
                        label = target['target_label']
                        seed = (20261011 + int(cid.split('-')[0]) * 10
                                + int(cid.split('-')[1]) + (0 if label == 'A' else 101))
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        np.random.seed(seed)
                        random.seed(seed)
                        torch.cuda.synchronize(device)
                        before = time.perf_counter()
                        wave, sample_rate = converter(source_file, target_files[label])
                        torch.cuda.synchronize(device)
                        infer_seconds = time.perf_counter() - before
                        wave = to24(wave, int(sample_rate))
                        before_score = time.perf_counter()
                        score = measurements(scorer_model, scorer, wave,
                                             source_scores[cid], target, device)
                        torch.cuda.synchronize(device)
                        score_seconds = time.perf_counter() - before_score
                        member = f'{cid}/{args.arm}_vc_{label}.mp3'
                        mp3, clipped = encode_mp3(wave)
                        add_bytes(output, member, mp3)
                        records.append({'arm': args.arm, 'case_id': cid,
                                        'source_uid': source['uid'],
                                        'target_uid': target['uid'],
                                        'target_label': label, 'emotion': source['emotion'],
                                        'seed': seed, 'mp3_member': member,
                                        'rl_training_uid_overlap': bool(source['rl_training_uid_overlap']),
                                        'inference_seconds': infer_seconds,
                                        'scoring_seconds': score_seconds,
                                        'clipped_fraction': clipped,
                                        'sample_rate_native': sample_rate, **score})
                        print(json.dumps({'event': 'EXTERNAL_PAIR_COMPLETE',
                                          'arm': args.arm, 'rank': rank,
                                          'case_id': cid, 'target': label,
                                          'inference_seconds': infer_seconds}), flush=True)
    os.replace(partial_tar, archive_path)
    write_parquet(stage / f'metrics-rank{rank}.parquet', records)
    atomic_json(stage / f'proof-rank{rank}.json',
                {'status': 'COMPLETE', 'rank': rank, 'arm': args.arm,
                 'predictions': len(records), 'model': metadata,
                 'packed_audio': str(archive_path)})
    dist.barrier()
    if rank == 0:
        all_rows = [row for worker in range(world)
                    for row in pq.read_table(stage / f'metrics-rank{worker}.parquet').to_pylist()]
        expected = 8 if args.smoke else 80
        assert len(all_rows) == expected
        assert len({(r['case_id'], r['target_label']) for r in all_rows}) == expected
        write_parquet(stage / 'metrics.parquet', all_rows)
        keys = ('audiobox_ce', 'audiobox_pq', 'top2_emotion_mse',
                'target_camp_cosine', 'inference_seconds', 'scoring_seconds')
        summary = {key: float(np.mean([row[key] for row in all_rows])) for key in keys}
        atomic_json(stage / 'DONE.json',
                    {'status': 'COMPLETE', 'arm': args.arm, 'smoke': args.smoke,
                     'sources': expected // 2, 'targets': 2,
                     'predictions': expected, 'packed_tars': 4,
                     'model': metadata, 'summary': summary,
                     'caveat': 'Same 40 curated inputs, two targets and Ears Medium/CAMPPlus proxies; not human MOS, independent quality, identity verification or WER.'})
        print(json.dumps({'event': 'EXTERNAL_EVAL_DONE', 'arm': args.arm,
                          'predictions': expected, 'summary': summary}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
