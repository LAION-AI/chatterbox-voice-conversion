import sys, json, time, glob, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"/tmp/vcbon"); import librosa, torch
from vc_batch import load_vc, set_target, convert_n
vc=load_vc("cuda:0"); set_target(vc,"/tmp/vcbon/target/reference.mp3")
srcs=sorted(glob.glob("/tmp/vcbon/src/*.mp3"))
durs=sorted([(f,librosa.get_duration(path=f)) for f in srcs],key=lambda x:x[1])
picks=[("short",durs[0]),("median",durs[len(durs)//2]),("long",durs[-1])]
convert_n(vc,picks[1][1][0],2)  # warmup
rows=[]
for tag,(f,d) in picks:
    base=None
    for B in (1,4,8,16,32):
        _,gen,od=convert_n(vc,f,B,seed=1234)
        per=gen/B*1000
        if B==1: base=gen
        rows.append({"clip":tag,"src_s":d,"B":B,"gen_s":gen,"per_cand_ms":per,
                     "rtf":gen/(B*od),"speedup":(base/(gen/B))})
json.dump({"rows":rows},open("/tmp/vcbon/out/rtf_sweep.json","w"),indent=2)
print("rtf_sweep rows:",len(rows))
