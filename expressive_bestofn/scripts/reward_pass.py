"""Reward pass: compute PEAK emotion intensity (max over all 40 emotion experts)
per candidate, for raw and SIDON pools, and cache every candidate mp3 so the
reducer can copy the reward-winner. seed=1234+local_idx (matches worker/refine).

Usage: CUDA_VISIBLE_DEVICES=<g> WHISPER_DIR/EMO_DIR/QUAL_DIR set; python reward_pass.py <shard> <nsh>
"""
import os, sys, json, glob, time, warnings, subprocess, tempfile
warnings.filterwarnings("ignore")
import numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0, "/tmp/vcbon")
from vc_batch import load_vc, set_target, convert_n
import worker as W
from refine_pass import load_sidon, sidon_restore

DEV = "cuda:0"; SRC_DIR = "/tmp/vcbon/src"; OUT = "/tmp/vcbon/out/reward"
ALL = "/tmp/vcbon/out/allcand"; TARGET = "/tmp/vcbon/target/reference.mp3"; N = 32


def to_mp3(wav, sr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); sf.write(tmp.name, wav, sr)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",tmp.name,"-ac","1","-b:a","128k",path], check=True)
    os.unlink(tmp.name)


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(OUT, exist_ok=True)
    vc = load_vc(DEV); set_target(vc, TARGET)
    fe, enc, emo, qual = W.load_scorer()
    sfe, sdec, spre = load_sidon()
    srcs = sorted(glob.glob(SRC_DIR + "/*.mp3"))
    mine = [(i, f) for i, f in enumerate(srcs) if i % nsh == shard]
    outp = os.path.join(OUT, f"shard{shard}.jsonl")
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)["id"])
            except: pass
    fout = open(outp, "a")
    print(f"[reward {shard}] {len(mine)} sources", flush=True)
    for si, (gi, f) in enumerate(mine):
        sid = os.path.basename(f)[:-4]
        if sid in done: continue
        t0 = time.time()
        wavs = convert_n(vc, f, N, seed=1234 + si)[0]
        c16 = [librosa.resample(w, orig_sr=vc.sr, target_sr=16000) for w in wavs]
        emb = W.embed(fe, enc, c16)
        raw40 = W.score_emotions(emb, emo, W.EMO)                  # dict emo->[N]
        raw_all = np.stack([raw40[e] for e in W.EMO], axis=1)      # [N,40]
        raw_peak = raw_all.max(axis=1)                            # [N]
        raw_ov = W.score_quality(emb, qual)["Overall_Quality"]
        # SIDON
        sid48 = [sidon_restore(w, vc.sr, sfe, sdec, spre) for w in wavs]
        s16 = [librosa.resample(w, orig_sr=48000, target_sr=16000) for w in sid48]
        semb = W.embed(fe, enc, s16)
        s40 = W.score_emotions(semb, emo, W.EMO)
        s_all = np.stack([s40[e] for e in W.EMO], axis=1)
        s_peak = s_all.max(axis=1)
        s_ov = W.score_quality(semb, qual)["Overall_Quality"]
        # cache all candidates for reducer to copy the winner
        d = os.path.join(ALL, sid)
        for i in range(N):
            to_mp3(wavs[i], vc.sr, os.path.join(d, f"raw_{i}.mp3"))
            to_mp3(sid48[i], 48000, os.path.join(d, f"sidon_{i}.mp3"))
        rec = {"id": sid,
               "raw_peak": [round(float(x), 4) for x in raw_peak],
               "raw_peak_emo": [W.EMO[int(j)] for j in raw_all.argmax(axis=1)],
               "raw_overall": [round(float(x), 4) for x in raw_ov],
               "sidon_peak": [round(float(x), 4) for x in s_peak],
               "sidon_peak_emo": [W.EMO[int(j)] for j in s_all.argmax(axis=1)],
               "sidon_overall": [round(float(x), 4) for x in s_ov]}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"[reward {shard}] {si+1}/{len(mine)} {sid[:32]:32s} "
              f"rawPeak {raw_peak.max():.2f} sidonPeak {s_peak.max():.2f} {time.time()-t0:.1f}s", flush=True)
    fout.close()
    print(f"[reward {shard}] DONE", flush=True)


if __name__ == "__main__":
    main()
