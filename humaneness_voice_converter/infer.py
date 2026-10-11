#!/usr/bin/env python3
"""Humaneness Voice Converter: one source clip + one target voice -> 24-kHz WAV.

The release contains the complete S3Gen weights and a frozen Humaneness Ears
Medium predictor for its 99 conditioning scores. A CUDA GPU is required for
the tested path. No transcript, TTS model, or external API is used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
from safetensors.torch import load_file
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch import nn
from transformers import WhisperFeatureExtractor

from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3gen.utils.mel import mel_spectrogram

TARGET_STATIC = {'AGEV', 'GEND', 'BRGT', 'HARM', 'ROUG', 'SMTH', 'WARM',
                 'R_CHST', 'R_HEAD', 'R_MASK', 'R_MIXD', 'R_NASL',
                 'R_ORAL', 'R_THRT'}


class StyledSpeakerProjection(nn.Module):
    """Exact score projection used by the 300h -> RL -> 40k training chain."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.style = nn.Sequential(nn.Linear(198, 256), nn.SiLU(),
                                   nn.Linear(256, 80))
        self.condition = None
        self.present = None
        self.scale = 1.0

    def forward(self, camp: torch.Tensor) -> torch.Tensor:
        if self.condition is None or self.present is None:
            raise RuntimeError('Set the 99 conditioning scores before inference')
        features = torch.cat((self.condition * self.present, self.present), dim=-1)
        return self.base(camp) + self.scale * self.style(features)


def resample(wave: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return wave.astype(np.float32, copy=False)
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(wave, target_rate // divisor,
                         source_rate // divisor).astype(np.float32)


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    wave, rate = sf.read(path, dtype='float32', always_2d=True)
    wave = wave.mean(axis=1)
    if not np.isfinite(wave).all() or not (1 <= len(wave) / rate <= 30):
        raise ValueError(f'{path}: expected finite 1–30 second audio')
    return wave, rate


def tokenize_waveform(tokenizer, wave16: np.ndarray) -> np.ndarray:
    """Bounded 29-second windows with four-second overlap, as in training."""
    window, stride = 29 * 16000, 25 * 16000
    segments = []
    start = 0
    while start < len(wave16):
        end = min(start + window, len(wave16))
        tokens, lengths = tokenizer([wave16[start:end]])
        segments.append(tokens[0, :int(lengths[0])].cpu().numpy().astype(np.int64))
        if end == len(wave16):
            break
        start += stride
    return np.concatenate([codes[0 if i == 0 else 50:
                                 len(codes) if i == len(segments) - 1 else -50]
                           for i, codes in enumerate(segments)])


def load_ears(directory: Path, device: torch.device):
    location = directory / 'model.py'
    spec = importlib.util.spec_from_file_location('humaneness_ears_release_model', location)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import frozen score predictor at {location}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = json.loads((directory / 'classes.json').read_text())['names']
    model = module.LayeredMultiTaskWhisper(directory / 'medium', len(classes),
                                          initialize_pretrained=False)
    model.load_state_dict(load_file(directory / 'medium/model.safetensors'), strict=True)
    model.to(device).eval().requires_grad_(False)
    extractor = WhisperFeatureExtractor.from_pretrained(directory / 'medium',
                                                        local_files_only=True)
    norm = json.loads((directory / 'training_normalization.json').read_text())
    return model, extractor, norm


def ears_raw_scores(waves24: list[np.ndarray], model, extractor, norm: dict,
                    device: torch.device) -> np.ndarray:
    waves16 = [resample(w, 24000, 16000) for w in waves24]
    features = extractor(waves16, sampling_rate=16000, padding='longest',
                         truncation=True, max_length=480000,
                         return_attention_mask=True, return_tensors='np')
    mel = features['input_features'].astype(np.float32)
    mask = features['attention_mask'].astype(np.int64)
    if mel.shape[-1] % 2:
        mel = np.pad(mel, ((0, 0), (0, 0), (0, 1)))
        mask = np.pad(mask, ((0, 0), (0, 1)))
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(torch.from_numpy(mel).to(device),
                       torch.from_numpy(mask).to(device), predict_events=False)
    raw = output['scores'].float().cpu().numpy()
    raw = raw * np.asarray(norm['score_std'], np.float32) + \
        np.asarray(norm['score_mean'], np.float32)
    assert raw.shape == (len(waves24), 192) and np.isfinite(raw).all()
    return raw


def ears_scores(waves24: list[np.ndarray], model, extractor, norm: dict,
                device: torch.device) -> np.ndarray:
    scores = ears_raw_scores(waves24, model, extractor, norm, device)[:, :99].copy()
    scores[:, :98] /= 6.0
    scores[:, 98] /= 10.0
    return scores


def load_converter(directory: Path, device: torch.device) -> S3Gen:
    model = S3Gen()
    model.flow.spk_embed_affine_layer = StyledSpeakerProjection(
        model.flow.spk_embed_affine_layer)
    model.load_state_dict(load_file(directory / 'model.safetensors'), strict=True)
    model.to(device).eval().requires_grad_(False)
    model.flow.decoder.training_cfg_rate = 0.0
    return model


def make_reference(model: S3Gen, target16: np.ndarray, target24: np.ndarray,
                   device: torch.device) -> dict:
    with torch.inference_mode():
        codes = tokenize_waveform(model.tokenizer, target16)
        length = min(len(codes), 250)  # exactly the training/evaluation prefix limit
        camp = model.speaker_encoder.inference(
            torch.from_numpy(target16[:160000].copy())[None].to(device)).flatten().float()
        mel = mel_spectrogram(torch.from_numpy(target24).to(device)[None]).squeeze(0).float()
    assert length >= 25 and camp.numel() == 192 and mel.shape[0] == 80
    frames = 2 * length
    if mel.shape[1] < frames:
        assert frames - mel.shape[1] <= 10
        mel = torch.nn.functional.pad(mel, (0, frames - mel.shape[1]), mode='replicate')
    mel = mel[:, :frames].T.contiguous()
    return {'prompt_token': torch.from_numpy(codes[:length].copy())[None].to(device),
            'prompt_token_len': torch.tensor([length], device=device),
            'prompt_feat': mel[None],
            'prompt_feat_len': torch.tensor([frames], device=device),
            'embedding': camp[None].to(device)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261011)
    parser.add_argument('--flow-steps', type=int, default=10)
    parser.add_argument('--candidates', type=int, choices=(1, 4, 8, 16), default=1,
                        help='Independent batched flow draws; N>1 ranks with CE/identity/emotion')
    parser.add_argument('--microbatch', type=int, choices=(1, 2, 4, 8, 16), default=16,
                        help='Maximum candidates per decoder/score batch')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16',
                        help='FP8 is intentionally not exposed: current tested path is slower')
    parser.add_argument('--metrics', type=Path, default=None,
                        help='Optional JSON with all candidate proxies, reward and timing')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the validated inference path')
    if not (1 <= args.flow_steps <= 50):
        raise ValueError('flow-steps must be 1–50; training/evaluation used 10')
    torch.set_num_threads(4)
    started = time.perf_counter()
    device = torch.device('cuda:0')
    source, source_rate = read_audio(args.source)
    target, target_rate = read_audio(args.target)
    source16 = resample(source, source_rate, 16000)
    source24 = resample(source, source_rate, 24000)
    target16 = resample(target, target_rate, 16000)
    target24 = resample(target, target_rate, 24000)
    model = load_converter(args.model_dir, device)
    ears_dir = args.model_dir / 'conditioning/ears-medium'
    ears_model, extractor, norm = load_ears(ears_dir, device)
    names = json.loads((args.model_dir / 'conditioning/score_order.json').read_text())['score_order']
    predicted_names = [item.removeprefix('emo_').removeprefix('vn_').removesuffix('_reg')
                       for item in norm['score_names'][:99]]
    assert len(names) == 99 and names == predicted_names
    source_target_raw = ears_raw_scores([source24, target24], ears_model,
                                        extractor, norm, device)
    scores = source_target_raw[0, :99].copy()
    scores[:98] /= 6.0
    scores[98] /= 10.0
    target_scores = source_target_raw[1, :99].copy()
    target_scores[:98] /= 6.0
    target_scores[98] /= 10.0
    for name in TARGET_STATIC:
        scores[names.index(name)] = target_scores[names.index(name)]
    projection = model.flow.spk_embed_affine_layer
    projection.condition = torch.from_numpy(scores[None]).to(device)
    projection.present = torch.ones((1, 99), device=device)
    reference = make_reference(model, target16, target24, device)
    tokens = tokenize_waveform(model.tokenizer, source16)
    if not (25 <= len(tokens) <= 750):
        raise ValueError('Source tokens outside tested 1–30 s range')
    token = torch.from_numpy(tokens[None]).to(device)
    prepared_seconds = time.perf_counter() - started
    waves = []
    generation_started = time.perf_counter()
    for offset in range(0, args.candidates, args.microbatch):
        count = min(args.microbatch, args.candidates - offset)
        # The first draw keeps the original single-sample seed. Changing
        # microbatch size changes later batch RNG assignments; the fixed
        # seed and fixed microbatch jointly define a reproducible run.
        torch.manual_seed(args.seed + 100003 * offset)
        torch.cuda.manual_seed_all(args.seed + 100003 * offset)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                    enabled=args.precision == 'bf16'):
            mel = model.flow_inference(speech_tokens=token.expand(count, -1).contiguous(),
                                       ref_dict=dict(reference),
                                       n_cfm_timesteps=args.flow_steps, finalize=True)
            generated, _ = model.hift_inference(mel)
        batch = generated.float().cpu().numpy()
        assert batch.shape[0] == count and np.isfinite(batch).all()
        waves.extend(batch[i].reshape(-1) for i in range(count))
        del mel, generated
    torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - generation_started
    scores_by_candidate = []
    winner_index = 0
    ranking_seconds = 0.0
    if args.candidates > 1:
        ranking_started = time.perf_counter()
        calibration = json.loads((args.model_dir /
                                  'conditioning/reward_calibration.json').read_text())
        terms = calibration['terms']
        top2 = np.argsort(source_target_raw[0, :40])[-2:]
        ce_index = norm['score_names'].index('audiobox:CE')
        pq_index = norm['score_names'].index('audiobox:PQ')
        output_raw = np.concatenate([
            ears_raw_scores([np.asarray(w, np.float32) for w in waves[i:i + 8]],
                            ears_model, extractor, norm, device)
            for i in range(0, len(waves), 8)], axis=0)
        target_camp = reference['embedding'].float().cpu().numpy().reshape(-1)
        target_camp /= max(float(np.linalg.norm(target_camp)), 1e-8)
        waves16 = np.stack([resample(np.asarray(w, np.float32), 24000, 16000)
                            for w in waves])
        with torch.inference_mode():
            camp = model.speaker_encoder.inference(
                torch.from_numpy(waves16).to(device)).float().cpu().numpy()
        camp /= np.maximum(np.linalg.norm(camp, axis=1, keepdims=True), 1e-8)
        for index, predicted in enumerate(output_raw):
            ce = float(predicted[ce_index])
            pq = float(predicted[pq_index])
            mse = float(np.mean((predicted[top2] - source_target_raw[0, top2]) ** 2))
            cosine = float(camp[index] @ target_camp)
            def z(name: str, value: float) -> float:
                return (value - terms[name]['mean']) / terms[name]['std']
            reward = (z('audiobox_ce', ce) + z('target_camp_cosine', cosine)
                      - z('top2_emotion_mse', mse)) / 3.0
            scores_by_candidate.append({'index': index, 'audiobox_ce': ce,
                                        'audiobox_pq': pq, 'top2_emotion_mse': mse,
                                        'target_camp_cosine': cosine,
                                        'selection_reward': reward})
        winner_index = max(range(len(scores_by_candidate)),
                           key=lambda i: (scores_by_candidate[i]['selection_reward'], -i))
        torch.cuda.synchronize(device)
        ranking_seconds = time.perf_counter() - ranking_started
    wave = waves[winner_index]
    assert len(wave) >= 24000 and np.isfinite(wave).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, np.clip(wave, -1.0, 1.0), 24000)
    report = {'status': 'COMPLETE', 'output': str(args.output),
                      'seconds': len(wave) / 24000, 'source_seconds': len(source) / source_rate,
                      'target_seconds': len(target) / target_rate,
                      'seed': args.seed, 'flow_steps': args.flow_steps,
                      'candidates': args.candidates, 'microbatch': args.microbatch,
                      'precision': args.precision, 'winner_index': winner_index,
                      'preparation_seconds': prepared_seconds,
                      'generation_seconds': generation_seconds,
                      'ranking_seconds': ranking_seconds,
                      'total_wall_seconds': time.perf_counter() - started,
                      'candidate_scores': scores_by_candidate,
                      'model': 'Humaneness Voice Converter 40k'}
    if args.metrics is not None:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
