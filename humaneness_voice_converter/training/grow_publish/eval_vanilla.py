#!/usr/bin/env python3
"""Score source clips and render CAMPPlus-only *unmodified* Chatterbox S3Gen.

Four independent Slurm ranks each write one TAR and one JSONL. Source clips,
reference voices, seeds and acoustic prompting match eval_followups.py. No
per-sample audio files are written. Quality and speaker-consistency estimates
are proxies for curation, not human MOS or diarization.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

import numpy as np
import pyarrow.parquet as pq
from safetensors.torch import load_file
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'grow_v1'))
from chatterbox.models.s3gen import S3Gen
from generate_listening import encode_mp3, ref_dict
from grow_publish.eval_followups import add_bytes, read_wave
from grow_v1.score_inputs import to16
from grow_v1.train_grow import EarsReward
from prepare import WEIGHTS

OUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval')
VANILLA = OUT / 'vanilla_curation'


def camp_window_min_cosine(model: S3Gen, wave: np.ndarray,
                           device: torch.device) -> float:
    """A weak multi-speaker screen: 2-s windows with 1-s stride."""
    audio = to16(wave, 24000)
    width, stride = 32000, 16000
    if len(audio) < width * 2:
        windows = [audio[:width], audio[-width:]]
    else:
        starts = list(range(0, len(audio) - width + 1, stride))
        if starts[-1] != len(audio) - width:
            starts.append(len(audio) - width)
        windows = [audio[start:start + width] for start in starts]
    vectors = []
    with torch.inference_mode():
        for window in windows:
            if len(window) < width:
                window = np.pad(window, (0, width - len(window)))
            vec = model.speaker_encoder.inference(
                torch.from_numpy(window.copy())[None].to(device)).float().cpu().numpy()[0]
            vec /= max(float(np.linalg.norm(vec)), 1e-8)
            vectors.append(vec)
    dots = np.asarray(vectors) @ np.asarray(vectors).T
    return float(np.min(dots[np.triu_indices(len(vectors), k=1)]))


def main() -> None:
    rank = int(os.environ['SLURM_PROCID'])
    world = int(os.environ['SLURM_NTASKS'])
    assert world == 4 and 0 <= rank < 4 and torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.device_count() == 1
                          else 'cuda:' + os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    proof = json.loads((OUT / 'prepared.COMPLETE.json').read_text())
    assert proof['status'] == 'COMPLETE' and proof['sources'] == 160
    rows = pq.read_table(OUT / 'prepared.parquet').to_pylist()
    sources = sorted([row for row in rows if row['role'] == 'source'],
                     key=lambda row: (int(row['case_id'].split('-')[0]),
                                      int(row['case_id'].split('-')[1])))
    targets = sorted([row for row in rows if row['role'] == 'target'],
                     key=lambda row: row['target_label'])
    assert len(sources) == 160 and len(targets) == 2
    # This is the frozen upstream S3Gen: no q/k/v/o LoRA modules and no
    # StyledSpeakerProjection, hence no 99-dimensional VoiceNet/EmoNet input.
    model = S3Gen()
    missing, unexpected = model.load_state_dict(load_file(WEIGHTS), strict=False)
    assert not unexpected and set(missing) <= {'tokenizer.window'}
    assert model.flow.spk_embed_affine_layer.__class__.__name__ != 'StyledSpeakerProjection'
    assert not any(module.__class__.__name__ == 'LoRALinear' for module in model.modules())
    model = model.to(device).eval().requires_grad_(False)
    model.flow.decoder.training_cfg_rate = 0.0
    scorer = EarsReward(device)
    names = scorer.norm['score_names']
    ce_idx = names.index('audiobox:CE')
    pq_idx = names.index('audiobox:PQ')
    mos_idx = names.index('dnsmos:P808_MOS')
    target_refs = {row['target_label']: ref_dict(row, device, acoustic_prompt=True)
                   for row in targets}
    VANILLA.mkdir(parents=True, exist_ok=True)
    tar_partial = VANILLA / f'vanilla-rank{rank}.tar.partial'
    json_partial = VANILLA / f'vanilla-rank{rank}.jsonl.partial'
    records = []
    with tarfile.open(OUT / 'site_inputs.tar', 'r') as originals, tarfile.open(tar_partial, 'w') as archive:
        for position, source in enumerate(sources):
            if position % world != rank:
                continue
            case_id = source['case_id']
            wave = read_wave(originals, source['audio_member'])
            scores = scorer.score(wave[None])[0]
            consistency = camp_window_min_cosine(model, wave, device)
            rec = {'case_id': case_id, 'uid': source['uid'], 'lang': source['lang'],
                   'origin': source['origin'], 'emotion': source['emotion'],
                   'source_audiobox_ce': float(scores[ce_idx]),
                   'source_audiobox_pq': float(scores[pq_idx]),
                   'source_predicted_p808_mos': float(scores[mos_idx]),
                   'source_camp_window_min_cosine': consistency,
                   'predictions': {}}
            token = torch.from_numpy(np.frombuffer(source['tokens'], dtype='<i2').astype(np.int64)[None]).to(device)
            seed = 20261011 + int(case_id.split('-')[0]) * 10 + int(case_id.split('-')[1])
            for target in targets:
                label = target['target_label']
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    mel = model.flow_inference(speech_tokens=token,
                                               ref_dict=dict(target_refs[label]),
                                               n_cfm_timesteps=10, finalize=True)
                    generated, _ = model.hift_inference(mel)
                output = generated.float().cpu().numpy().reshape(-1)
                assert np.isfinite(output).all() and len(output) >= 24000
                member = f'{case_id}/vanilla_vc_{label}.mp3'
                mp3, clipped = encode_mp3(output)
                add_bytes(archive, member, mp3)
                target_camp = np.frombuffer(target['camp'], dtype='<f2').astype(np.float32)
                target_camp /= max(float(np.linalg.norm(target_camp)), 1e-8)
                with torch.inference_mode():
                    camp = model.speaker_encoder.inference(
                        torch.from_numpy(to16(output, 24000).copy())[None].to(device)).float().cpu().numpy()[0]
                camp /= max(float(np.linalg.norm(camp)), 1e-8)
                out_scores = scorer.score(output[None])[0]
                top2 = np.argsort(scores[:40])[-2:]
                rec['predictions'][label] = {
                    'mp3_member': member, 'seed': seed,
                    'audiobox_ce': float(out_scores[ce_idx]),
                    'audiobox_pq': float(out_scores[pq_idx]),
                    'top2_emotion_mse': float(np.mean((out_scores[top2] - scores[top2]) ** 2)),
                    'target_camp_cosine': float(camp @ target_camp),
                    'clipped_fraction': clipped,
                    'duration_s': len(output) / 24000,
                }
            records.append(rec)
            if len(records) % 10 == 0:
                print(json.dumps({'rank': rank, 'source_cases': len(records)}), flush=True)
    with json_partial.open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + '\n')
    os.replace(tar_partial, VANILLA / f'vanilla-rank{rank}.tar')
    os.replace(json_partial, VANILLA / f'vanilla-rank{rank}.jsonl')
    proof = {'status': 'COMPLETE', 'rank': rank, 'sources': len(records),
             'predictions': len(records) * 2, 'weights_sha256': hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
             'vanilla_no_lora': True, 'vanilla_no_style_projection': True,
             'new_per_sample_audio_files': 0}
    (VANILLA / f'vanilla-rank{rank}.COMPLETE.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
