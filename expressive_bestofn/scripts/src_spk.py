"""Source->target speaker-sim baseline (ECAPA + Orange) for all sources. One GPU."""
import os, sys, json, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F, librosa, huggingface_hub
sys.path.insert(0, "/home/deployer/laion/Voice-Acting-Pipeline/scripts")
DEV="cuda:0"
_o=huggingface_hub.hf_hub_download
huggingface_hub.hf_hub_download=lambda *a,**k:(k.pop("use_auth_token",None),_o(*a,**k))[1]
from speechbrain.inference.speaker import EncoderClassifier
ecapa=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",savedir="/tmp/vcbon/ecapa_ckpt",run_opts={"device":DEV})
from spk_embeddings import EmbeddingsModel
EmbeddingsModel.all_tied_weights_keys={}
orange=EmbeddingsModel.from_pretrained("Orange/Speaker-wavLM-tbr").to(DEV).eval()
def ec(w):
    with torch.no_grad(): e=ecapa.encode_batch(torch.from_numpy(np.asarray(w,np.float32)).to(DEV)[None,:]).squeeze()
    return F.normalize(e,dim=0)
def orr(w):
    with torch.no_grad(): return orange(torch.from_numpy(np.asarray(w,np.float32)).to(DEV)[None,:]).squeeze(0)
tw,_=librosa.load("/tmp/vcbon/target/reference.mp3",sr=16000,mono=True); tw=tw*(0.97/(np.abs(tw).max()+1e-9))
tec=ec(tw); tor=orr(tw)
out={}
for f in sorted(glob.glob("/tmp/vcbon/src/*.mp3")):
    sid=os.path.basename(f)[:-4]; w,_=librosa.load(f,sr=16000,mono=True,duration=30)
    out[sid]={"ecapa":round(float((ec(w)*tec).sum()),4),"orange":round(float((orr(w)*tor).sum()),4)}
json.dump(out,open("/tmp/vcbon/out/src_spk.json","w"),indent=2)
print("src_spk baseline for",len(out),"sources -> ECAPA mean",round(np.mean([v['ecapa'] for v in out.values()]),3),
      "Orange mean",round(np.mean([v['orange'] for v in out.values()]),3))
