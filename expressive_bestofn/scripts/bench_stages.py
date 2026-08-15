import os,sys,time,glob,warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, librosa, torchaudio, torch.nn.functional as F, huggingface_hub
sys.path.insert(0,"/tmp/vcbon"); sys.path.insert(0,"/home/deployer/laion/Voice-Acting-Pipeline/scripts")
from vc_batch import load_vc,set_target,convert_n
import worker as W
from refine_pass import load_sidon, sidon_restore
DEV="cuda:0"
vc=load_vc(DEV); set_target(vc,"/tmp/vcbon/target/reference.mp3")
fe,enc,emo,qual=W.load_scorer()
sfe,sdec,spre=load_sidon()
srcs=sorted(glob.glob("/tmp/vcbon/src/*.mp3"))
durs=[(f,librosa.get_duration(path=f)) for f in srcs]; import statistics
meddur=statistics.median(d for _,d in durs); f=min(durs,key=lambda x:abs(x[1]-meddur))[0]
B=32
convert_n(vc,f,4)  # warmup
# gen
torch.cuda.synchronize(); t=time.time()
wavs,gen_s,od=convert_n(vc,f,B,seed=1); torch.cuda.synchronize()
tgen=time.time()-t
# sidon (32)
t=time.time()
sid=[sidon_restore(w,vc.sr,sfe,sdec,spre) for w in wavs]; torch.cuda.synchronize()
tsid=time.time()-t
# resample 32 to 16k
c16=[librosa.resample(w,orig_sr=vc.sr,target_sr=16000) for w in wavs]
# score: whisper encode + 40 emo + 4 qual on 32
t=time.time()
emb=W.embed(fe,enc,c16); _=W.score_emotions(emb,emo,W.EMO); _=W.score_quality(emb,qual); torch.cuda.synchronize()
tsc=time.time()-t
print(f"CLIP dur={meddur:.1f}s out={od:.1f}s  B={B}")
print(f"gen32   {tgen:.2f}s  {tgen/B*1000:.0f} ms/cand")
print(f"sidon32 {tsid:.2f}s  {tsid/B*1000:.0f} ms/cand")
print(f"score32 {tsc:.2f}s  {tsc/B*1000:.0f} ms/cand")
print(f"TOTAL/cand gen+sidon+score = {(tgen+tsid+tsc)/B*1000:.0f} ms")
import json; json.dump({"dur":meddur,"out":od,"B":B,"tgen":tgen,"tsid":tsid,"tsc":tsc,
 "gen_ms":tgen/B*1000,"sid_ms":tsid/B*1000,"sc_ms":tsc/B*1000},open("/tmp/vcbon/out/stage_bench.json","w"),indent=2)
