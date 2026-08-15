"""Unified refinement pass (one GPU).

For each source, regenerate the SAME 32 Chatterbox candidates (seed=1234+local_idx,
identical to worker.py) and:
  (a) RAW speaker-similarity to target: ECAPA-TDNN (192-d) + Orange/Speaker-wavLM-tbr (128-d).
  (b) SIDON post-process each candidate (sarulab-speech/sidon-v0.1, 48 kHz restoration),
      then re-score the refined audio: emotion top-3 (cos + MAE), quality experts,
      and ECAPA + Orange speaker-sim to target.
Emotion/quality of the RAW pool already live in worker.py's records (reused by reducer).

Usage: CUDA_VISIBLE_DEVICES=<g> WHISPER_DIR/EMO_DIR/QUAL_DIR set; python refine_pass.py <shard> <nsh>
"""
import os, sys, json, glob, time, warnings, subprocess, tempfile
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F, librosa, torchaudio, soundfile as sf
import huggingface_hub

sys.path.insert(0, "/tmp/vcbon")
sys.path.insert(0, "/home/deployer/laion/Voice-Acting-Pipeline/scripts")
from vc_batch import load_vc, set_target, convert_n
import worker as W  # reuses scorer (needs WHISPER_DIR/EMO_DIR/QUAL_DIR env at import)

DEV = "cuda:0"
SRC_DIR = "/tmp/vcbon/src"
OUT = "/tmp/vcbon/out/refine"
TARGET = "/tmp/vcbon/target/reference.mp3"
N_CAND = 32

_orig = huggingface_hub.hf_hub_download
def _patched(*a, **k): k.pop("use_auth_token", None); return _orig(*a, **k)
huggingface_hub.hf_hub_download = _patched


def load_spk():
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir="/tmp/vcbon/ecapa_ckpt", run_opts={"device": DEV})
    from spk_embeddings import EmbeddingsModel
    EmbeddingsModel.all_tied_weights_keys = {}
    orange = EmbeddingsModel.from_pretrained("Orange/Speaker-wavLM-tbr").to(DEV).eval()
    return ecapa, orange


def load_sidon():
    import transformers
    fe_path = huggingface_hub.hf_hub_download("sarulab-speech/sidon-v0.1", filename="feature_extractor_cuda.pt")
    dec_path = huggingface_hub.hf_hub_download("sarulab-speech/sidon-v0.1", filename="decoder_cuda.pt")
    fe = torch.jit.load(fe_path, map_location=DEV).to(DEV)
    decoder = torch.jit.load(dec_path, map_location=DEV).to(DEV)
    pre = transformers.SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0", sampling_rate=16000)
    return fe, decoder, pre


@torch.inference_mode()
def sidon_restore(wav, sr, fe, decoder, pre):
    """wav: 1-D np float. Returns restored 48kHz np float (mirrors run_full_enhanced_eval)."""
    w = torch.from_numpy(np.asarray(wav, np.float32)).view(1, -1)
    peak = w.abs().max()
    if peak > 0: w = 0.9 * (w / peak)
    target_n = int(48000 / sr * w.shape[-1])
    w16 = torchaudio.functional.highpass_biquad(w, sr, 50)
    w16 = torchaudio.functional.resample(w16, sr, 16000)
    w16 = F.pad(w16, (0, 24000))
    restoreds, cache = [], None
    for chunk in w16.view(-1).split(16000 * 96):
        inputs = pre(F.pad(chunk, (160, 160)), return_tensors="pt")
        feature = fe(inputs["input_features"].to(DEV))["last_hidden_state"]
        if cache is not None: feature = torch.cat([cache, feature], dim=1)
        restoreds.append(decoder(feature.transpose(1, 2)).view(-1)[:-960])
        cache = feature[:, -1:]
    return torch.cat(restoreds, dim=0)[:target_n].cpu().numpy()


@torch.no_grad()
def ecapa_emb(ecapa, w16):
    e = ecapa.encode_batch(torch.from_numpy(np.asarray(w16, np.float32)).to(DEV)[None, :]).squeeze()
    return F.normalize(e, dim=0)

@torch.no_grad()
def orange_emb(orange, w16):
    return orange(torch.from_numpy(np.asarray(w16, np.float32)).to(DEV)[None, :]).squeeze(0)


AUD = "/tmp/vcbon/out/audio"
def to_mp3(wav, sr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); sf.write(tmp.name, wav, sr)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",tmp.name,"-ac","1","-b:a","128k",path], check=True)
    os.unlink(tmp.name)


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(OUT, exist_ok=True)
    # source -> {top3, src_top3, raw per-candidate emo_mae/emo_cos/Overall/Speech}
    top3map, rawmap = {}, {}
    for p in glob.glob("/tmp/vcbon/out/scores/shard*.jsonl"):
        for l in open(p):
            r = json.loads(l); top3map[r["id"]] = r["top3_emotions"]
            top3map[r["id"]+"__src"] = r["src_top3"]
            rawmap[r["id"]] = {c["i"]: {"emo_mae": c["emo_mae"], "emo_cos": c["emo_cos"],
                               "Overall_Quality": c["quality"]["Overall_Quality"],
                               "Speech_Quality": c["quality"]["Speech_Quality"]} for c in r["cands"]}
    vc = load_vc(DEV); set_target(vc, TARGET)
    fe_w, enc, emo_experts, qual_experts = W.load_scorer()
    ecapa, orange = load_spk()
    sfe, sdec, spre = load_sidon()
    tw, _ = librosa.load(TARGET, sr=16000, mono=True); tw = tw * (0.97 / (np.abs(tw).max() + 1e-9))
    t_ec = ecapa_emb(ecapa, tw); t_or = orange_emb(orange, tw)
    srcs = sorted(glob.glob(SRC_DIR + "/*.mp3"))
    mine = [(i, f) for i, f in enumerate(srcs) if i % nsh == shard]
    outp = os.path.join(OUT, f"shard{shard}.jsonl")
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)["id"])
            except: pass
    fout = open(outp, "a")
    print(f"[refine {shard}] {len(mine)} sources", flush=True)
    for si, (gi, f) in enumerate(mine):
        sid = os.path.basename(f)[:-4]
        if sid in done: continue
        t0 = time.time()
        top3 = top3map[sid]; src_top3 = np.array(top3map[sid + "__src"])
        wavs24, _, _ = convert_n(vc, f, N_CAND, seed=1234 + si)
        # raw speaker sim + SIDON refine
        raw_ec, raw_or = [], []
        sid_wavs16, sid_wavs48 = [], []
        for w in wavs24:
            w16 = librosa.resample(w, orig_sr=vc.sr, target_sr=16000)
            raw_ec.append(float((ecapa_emb(ecapa, w16) * t_ec).sum()))
            raw_or.append(float((orange_emb(orange, w16) * t_or).sum()))
            sw48 = sidon_restore(w, vc.sr, sfe, sdec, spre)
            sid_wavs48.append(sw48)
            sid_wavs16.append(librosa.resample(sw48, orig_sr=48000, target_sr=16000))
        # SIDON scoring — batch whisper encode over 32
        cemb = W.embed(fe_w, enc, sid_wavs16)
        ce = W.score_emotions(cemb, emo_experts, top3)   # dict emo->[N]
        cq = W.score_quality(cemb, qual_experts)          # dict lab->[N]
        cand_top3 = np.stack([ce[e] for e in top3], axis=1)  # [N,3]
        s_cos = (cand_top3 @ src_top3) / (np.linalg.norm(cand_top3, axis=1) * (np.linalg.norm(src_top3) + 1e-9) + 1e-9)
        s_mae = np.abs(cand_top3 - src_top3[None, :]).mean(axis=1)
        sid_ec, sid_or = [], []
        for w16 in sid_wavs16:
            sid_ec.append(float((ecapa_emb(ecapa, w16) * t_ec).sum()))
            sid_or.append(float((orange_emb(orange, w16) * t_or).sum()))
        sidon = []
        for i in range(N_CAND):
            sidon.append({"i": i, "emo_cos": round(float(s_cos[i]), 4), "emo_mae": round(float(s_mae[i]), 4),
                          "quality": {lab: round(float(cq[lab][i]), 4) for lab in cq},
                          "ecapa": round(sid_ec[i], 4), "orange": round(sid_or[i], 4)})
        # ---- pick re-ranked candidates for the grid + save audio ----
        rawc = rawmap[sid]
        raw_bestQ = max(range(N_CAND), key=lambda i: rawc[i]["Overall_Quality"])
        raw_bestMAE = min(range(N_CAND), key=lambda i: rawc[i]["emo_mae"])
        sid_bestQ = max(range(N_CAND), key=lambda i: sidon[i]["quality"]["Overall_Quality"])
        sid_bestMAE = min(range(N_CAND), key=lambda i: sidon[i]["emo_mae"])
        d = os.path.join(AUD, sid)
        to_mp3(wavs24[raw_bestQ], vc.sr, os.path.join(d, "raw_bestQ.mp3"))
        to_mp3(wavs24[raw_bestMAE], vc.sr, os.path.join(d, "raw_bestMAE.mp3"))
        to_mp3(sid_wavs48[sid_bestQ], 48000, os.path.join(d, "sidon_bestQ.mp3"))
        to_mp3(sid_wavs48[sid_bestMAE], 48000, os.path.join(d, "sidon_bestMAE.mp3"))
        def pick(kind, i, sc):
            return {"kind": kind, "i": int(i), "emo_cos": sc["emo_cos"], "emo_mae": sc["emo_mae"],
                    "Overall_Quality": sc["Overall_Quality"], "Speech_Quality": sc["Speech_Quality"],
                    "ecapa": sc["ecapa"], "orange": sc["orange"]}
        def rawsc(i): return {**rawc[i], "ecapa": raw_ec[i], "orange": raw_or[i]}
        def sidsc(i): return {"emo_cos": sidon[i]["emo_cos"], "emo_mae": sidon[i]["emo_mae"],
                              "Overall_Quality": sidon[i]["quality"]["Overall_Quality"],
                              "Speech_Quality": sidon[i]["quality"]["Speech_Quality"],
                              "ecapa": sidon[i]["ecapa"], "orange": sidon[i]["orange"]}
        picks = {"raw_bestQ": pick("raw_bestQ", raw_bestQ, rawsc(raw_bestQ)),
                 "raw_bestMAE": pick("raw_bestMAE", raw_bestMAE, rawsc(raw_bestMAE)),
                 "sidon_bestQ": pick("sidon_bestQ", sid_bestQ, sidsc(sid_bestQ)),
                 "sidon_bestMAE": pick("sidon_bestMAE", sid_bestMAE, sidsc(sid_bestMAE))}
        rec = {"id": sid, "raw_ecapa": [round(x, 4) for x in raw_ec], "raw_orange": [round(x, 4) for x in raw_or],
               "sidon": sidon, "picks": picks}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"[refine {shard}] {si+1}/{len(mine)} {sid[:32]:32s} "
              f"rawECAPA {np.mean(raw_ec):.3f}->sidon {np.mean(sid_ec):.3f} | "
              f"rawMAE(worker) sidonMAE {s_mae.mean():.3f} | {time.time()-t0:.1f}s", flush=True)
    fout.close()
    print(f"[refine {shard}] DONE", flush=True)


if __name__ == "__main__":
    main()
