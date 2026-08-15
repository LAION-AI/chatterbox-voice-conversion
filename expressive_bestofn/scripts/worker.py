"""Fused generate+score worker (one GPU).

For each assigned source clip:
  1. Score the SOURCE with BUD-E-Whisper + 40 emotion experts -> top-3 emotions.
  2. Generate N=32 batched multi-seed voice conversions to the target voice.
  3. Score every candidate: the source's top-3 emotion dims + 4 quality experts.
  4. emotion-similarity = cosine of source-vs-candidate on the top-3 dims (+ MAE).
     speech-quality     = DNSMOS-distilled Overall/Speech quality expert.
  5. Write per-candidate scores JSONL; save the source + top-3 candidates as MP3.

Usage: CUDA_VISIBLE_DEVICES=<g> python worker.py <shard_idx> <n_shards>
"""
import os, sys, json, glob, time, warnings, subprocess, tempfile
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, librosa, soundfile as sf

sys.path.insert(0, "/tmp/vcbon")
from vc_batch import load_vc, set_target, convert_n

DEV = "cuda:0"
WHISPER_DIR = os.environ["WHISPER_DIR"]
EMO_DIR = os.environ["EMO_DIR"]
QUAL_DIR = os.environ["QUAL_DIR"]
SRC_DIR = "/tmp/vcbon/src"
OUT_AUDIO = "/tmp/vcbon/out/audio"
OUT_SCORES = "/tmp/vcbon/out/scores"
N_CAND = 32
SR = 16000
SEQ_LEN, EMB, PROJ = 1500, 768, 64
HID, DROP = [64, 32, 16], [0.0, 0.1, 0.1, 0.1]

EMO = ["Affection","Amusement","Anger","Astonishment_Surprise","Awe","Bitterness","Concentration","Confusion",
"Contemplation","Contempt","Contentment","Disappointment","Disgust","Distress","Doubt","Elation","Embarrassment",
"Emotional_Numbness","Fatigue_Exhaustion","Fear","Helplessness","Hope_Enthusiasm_Optimism","Impatience_and_Irritability",
"Infatuation","Interest","Intoxication_Altered_States_of_Consciousness","Jealousy_&_Envy","Longing","Malevolence_Malice",
"Pain","Pleasure_Ecstasy","Pride","Relief","Sadness","Sexual_Lust","Shame","Sourness","Teasing","Thankfulness_Gratitude","Triumph"]
QUALITY = {"Overall_Quality":"model_score_overall_quality_best.pth",
           "Speech_Quality":"model_score_speech_quality_best.pth",
           "Background_Quality":"model_score_background_quality_best.pth",
           "Content_Enjoyment":"model_score_content_enjoyment_best.pth"}


class FullEmbeddingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(SEQ_LEN*EMB, PROJ)
        layers = [nn.ReLU(), nn.Dropout(DROP[0])]; cur = PROJ
        for i, h in enumerate(HID):
            layers += [nn.Linear(cur, h), nn.ReLU(), nn.Dropout(DROP[i+1])]; cur = h
        layers.append(nn.Linear(cur, 1)); self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        if x.ndim == 4 and x.shape[1] == 1: x = x.squeeze(1)
        return self.mlp(self.proj(self.flatten(x)))


class PooledEmbeddingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3072, PROJ)
        layers = [nn.ReLU(), nn.Dropout(DROP[0])]; cur = PROJ
        for i, h in enumerate(HID):
            layers += [nn.Linear(cur, h), nn.ReLU(), nn.Dropout(DROP[i+1])]; cur = h
        layers.append(nn.Linear(cur, 1)); self.mlp = nn.Sequential(*layers)
    def forward(self, x): return self.mlp(self.proj(x))


def _clean_sd(sd):
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


def load_scorer():
    from transformers import WhisperModel, WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_DIR)
    wm = WhisperModel.from_pretrained(WHISPER_DIR, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    enc = wm.get_encoder().to(DEV).eval()
    # map emotion name -> file (filenames sanitize & and special chars)
    emo_experts = {}
    files = {os.path.basename(p): p for p in glob.glob(EMO_DIR + "/model_*_best.pth")}
    def find_file(name):
        for cand in (f"model_{name}_best.pth",
                     f"model_{name.replace('&','and')}_best.pth",
                     f"model_{name.replace('&','_and_').replace('__','_')}_best.pth"):
            if cand in files: return files[cand]
        # loose match
        key = name.lower().replace("&","and").replace("_","")
        for b, p in files.items():
            if b[6:-9].lower().replace("&","and").replace("_","") == key: return p
        return None
    for e in EMO:
        p = find_file(e)
        if p is None: raise RuntimeError(f"missing emotion expert for {e}")
        m = FullEmbeddingMLP().to(DEV)
        m.load_state_dict(_clean_sd(torch.load(p, map_location=DEV, weights_only=True)))
        m.eval(); m.half(); emo_experts[e] = m
    qual_experts = {}
    for label, fn in QUALITY.items():
        p = os.path.join(QUAL_DIR, fn)
        m = PooledEmbeddingMLP().to(DEV)
        m.load_state_dict(_clean_sd(torch.load(p, map_location=DEV, weights_only=True)))
        m.eval(); m.half(); qual_experts[label] = m
    return fe, enc, emo_experts, qual_experts


@torch.no_grad()
def embed(fe, enc, wavs16):
    inp = fe(wavs16, sampling_rate=SR, return_tensors="pt", padding="max_length", truncation=True)
    feats = inp.input_features.to(DEV, dtype=torch.float16)
    return enc(feats, return_dict=True).last_hidden_state  # [B,1500,768] fp16


@torch.no_grad()
def score_emotions(emb, emo_experts, names):
    return {e: emo_experts[e](emb).squeeze(-1).float().cpu().numpy() for e in names}  # each [B]


@torch.no_grad()
def score_quality(emb, qual_experts):
    e = emb.float()
    pooled = torch.cat([e.mean(1), e.min(1).values, e.max(1).values, e.std(1)], dim=1).half()  # [B,3072]
    return {lab: qual_experts[lab](pooled).squeeze(-1).float().cpu().numpy() for lab in qual_experts}


def to_mp3(wav, sr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); sf.write(tmp.name, wav, sr)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",tmp.name,"-ac","1","-b:a","128k",path], check=True)
    os.unlink(tmp.name)


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(OUT_AUDIO, exist_ok=True); os.makedirs(OUT_SCORES, exist_ok=True)
    vc = load_vc(DEV); set_target(vc, "/tmp/vcbon/target/reference.mp3")
    fe, enc, emo_experts, qual_experts = load_scorer()
    srcs = sorted(glob.glob(SRC_DIR + "/*.mp3"))
    mine = [f for i, f in enumerate(srcs) if i % nsh == shard]
    outp = os.path.join(OUT_SCORES, f"shard{shard}.jsonl")
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)["id"])
            except: pass
    fout = open(outp, "a")
    print(f"[shard {shard}] {len(mine)} sources", flush=True)
    for si, f in enumerate(mine):
        sid = os.path.basename(f)[:-4]
        if sid in done: continue
        t0 = time.time()
        # 1. source emotion top-3
        src16, _ = librosa.load(f, sr=SR, mono=True, duration=30)
        semb = embed(fe, enc, [src16])
        sall = {e: float(v[0]) for e, v in score_emotions(semb, emo_experts, EMO).items()}
        top3 = sorted(sall, key=sall.get, reverse=True)[:3]
        src_top3 = np.array([sall[e] for e in top3])
        squal = {lab: float(v[0]) for lab, v in score_quality(semb, qual_experts).items()}
        # 2. generate candidates
        wavs24, gen_s, out_dur = convert_n(vc, f, N_CAND, seed=1234 + si)
        # 3. score candidates (resample to 16k)
        c16 = [librosa.resample(w, orig_sr=vc.sr, target_sr=SR) for w in wavs24]
        cemb = embed(fe, enc, c16)
        ce = score_emotions(cemb, emo_experts, top3)      # dict emo->[N]
        cq = score_quality(cemb, qual_experts)            # dict lab->[N]
        cand_top3 = np.stack([ce[e] for e in top3], axis=1)  # [N,3]
        # metrics
        cos = (cand_top3 @ src_top3) / (np.linalg.norm(cand_top3, axis=1) * (np.linalg.norm(src_top3) + 1e-9) + 1e-9)
        mae = np.abs(cand_top3 - src_top3[None, :]).mean(axis=1)
        emo_sim = cos  # primary
        overall = cq["Overall_Quality"]; speechq = cq["Speech_Quality"]
        cands = []
        for i in range(N_CAND):
            cands.append({"i": i,
                          "emo3": {e: round(float(ce[e][i]), 4) for e in top3},
                          "emo_cos": round(float(cos[i]), 4), "emo_mae": round(float(mae[i]), 4),
                          "quality": {lab: round(float(cq[lab][i]), 4) for lab in cq}})
        # combined rank within group: z-normalized emo_sim + z-normalized overall quality
        def z(a):
            a = np.asarray(a, float); s = a.std();  return (a - a.mean()) / s if s > 1e-6 else a * 0
        combined = z(emo_sim) + z(overall)
        order = list(np.argsort(-combined))
        top_idx = order[:3]
        # save audio: source + top-3 candidates
        srcmp3 = os.path.join(OUT_AUDIO, sid, "source.mp3"); to_mp3(src16, SR, srcmp3)
        for rank, i in enumerate(top_idx):
            to_mp3(wavs24[i], vc.sr, os.path.join(OUT_AUDIO, sid, f"top{rank+1}.mp3"))
        rec = {"id": sid, "src_dur": round(len(src16)/SR, 2), "out_dur": round(out_dur, 2),
               "gen_s_32": round(gen_s, 3), "top3_emotions": top3, "src_top3": [round(float(x),4) for x in src_top3],
               "src_all_emo": {e: round(v,4) for e, v in sall.items()},
               "src_quality": {k: round(v,4) for k, v in squal.items()},
               "cands": cands, "top_idx": [int(x) for x in top_idx],
               "combined": [round(float(x),4) for x in combined]}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"[shard {shard}] {si+1}/{len(mine)} {sid[:36]:36s} top3={top3} "
              f"bestcos={emo_sim.max():.3f} bestQ={overall.max():.2f} {time.time()-t0:.1f}s", flush=True)
    fout.close()
    print(f"[shard {shard}] DONE", flush=True)


if __name__ == "__main__":
    main()
