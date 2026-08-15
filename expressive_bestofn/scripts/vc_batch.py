"""Batched multi-seed Chatterbox voice conversion.

One (source, target) pair -> B candidate conversions in a SINGLE forward pass.
Variation comes from the CFM decoder's Gaussian noise init: a batched
randn_like(mu) of shape (B,80,T) draws B independent noises -> B distinct outputs,
equivalent to B different seeds but far faster than B sequential calls.
"""
import time, librosa, numpy as np, torch
from chatterbox.tts import ChatterboxTTS  # noqa (ensures pkg import path)
from chatterbox.vc import ChatterboxVC
from chatterbox.models.s3tokenizer import S3_SR


def load_vc(device):
    return ChatterboxVC.from_pretrained(device)


def set_target(vc, target_path):
    """Normalize target to peak 0.97 then set as target voice (ref_dict)."""
    w, sr = librosa.load(target_path, sr=None)
    peak = np.abs(w).max()
    if peak > 0:
        w = w * (0.97 / peak)
    import soundfile as sf, tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, w, sr)
    vc.set_target_voice(tmp.name)
    os.unlink(tmp.name)


@torch.inference_mode()
def convert_n(vc, source_path, n, seed=0):
    """Return (wavs_list[np.float32 @24k], gen_seconds, out_dur_seconds).

    Generates n candidate conversions of source_path to the pre-set target
    in one batched forward pass.
    """
    s3gen = vc.s3gen
    dev = vc.device
    audio_16, _ = librosa.load(source_path, sr=S3_SR)
    audio_16 = torch.from_numpy(audio_16).float().to(dev)[None, ]
    s3_tokens, _ = s3gen.tokenizer(audio_16)          # (1, T)
    tok = s3_tokens.repeat(n, 1).contiguous()         # (n, T)
    torch.manual_seed(seed)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    wavs, _ = s3gen.inference(speech_tokens=tok, ref_dict=vc.ref_dict)  # (n, samples)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    gen_s = time.time() - t0
    wavs = wavs.detach().cpu().float().numpy()
    out = [wavs[i] for i in range(wavs.shape[0])]
    out_dur = out[0].shape[-1] / s3gen.mel2wav.sampling_rate if hasattr(s3gen.mel2wav, "sampling_rate") else out[0].shape[-1] / 24000
    return out, gen_s, out_dur


if __name__ == "__main__":
    import sys, glob, os
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    vc = load_vc(dev)
    set_target(vc, "/tmp/vcbon/target/reference.mp3")
    srcs = sorted(glob.glob("/tmp/vcbon/src/*.mp3"))
    # pick 3 representative durations: short, mid, long
    durs = [(f, librosa.get_duration(path=f)) for f in srcs]
    durs.sort(key=lambda x: x[1])
    picks = [durs[0], durs[len(durs)//2], durs[-1]]
    print("device:", dev, "| SR out:", vc.sr)
    # warmup
    convert_n(vc, picks[1][0], 2)
    print(f"\n{'clip':40s} {'src_s':>6s} {'B':>3s} {'gen_s':>7s} {'s/cand':>7s} {'RTF':>6s}")
    for f, d in picks:
        name = os.path.basename(f)[:38]
        for B in (1, 4, 8, 16, 32):
            wavs, gen_s, out_dur = convert_n(vc, f, B, seed=1234)
            per = gen_s / B
            rtf = gen_s / (B * out_dur)   # gen-time per second of audio produced
            print(f"{name:40s} {d:6.2f} {B:3d} {gen_s:7.2f} {per:7.3f} {rtf:6.3f}")
        # variation sanity: are the 32 candidates different?
        wavs, _, _ = convert_n(vc, f, 8, seed=1234)
        L = min(len(w) for w in wavs)
        arr = np.stack([w[:L] for w in wavs])
        pair = np.mean([np.corrcoef(arr[i], arr[j])[0,1] for i in range(4) for j in range(i+1,4)])
        print(f"   -> mean pairwise waveform corr among candidates: {pair:.3f} (lower=more variation)")
