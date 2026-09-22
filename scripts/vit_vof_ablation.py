from __future__ import annotations
from pathlib import Path
import sys, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))

from _data import load_vendored
from mprc_structural.features import (
    log_batch, fit_z256_scale, to_z256,
    feature_pixels, feature_adi9, feature_walk7,
    feature_vof_second_diff, feature_s5
)
from mprc_structural.lut import CategoricalLUT

X,y,train_idx,test_idx=load_vendored(ROOT)
tr,te=X[train_idx],X[test_idx]
ytr,yte=y[train_idx],y[test_idx]
log_tr=log_batch(tr,0.8)
log_te=log_batch(te,0.8)
scale=fit_z256_scale(log_tr)
Ztr,Zte=to_z256(log_tr,scale),to_z256(log_te,scale)

families={
    "pixel_control":feature_pixels,
    "adi9":feature_adi9,
    "walk7":feature_walk7,
    "vof_d2":feature_vof_second_diff,
    "s5":feature_s5,
}
train_feats={n:[fn(z) for z in Ztr] for n,fn in families.items()}
test_feats={n:[fn(z) for z in Zte] for n,fn in families.items()}
tables={}
for n,rows in train_feats.items():
    lut=CategoricalLUT(len(rows[0]),10)
    for row,label in zip(rows,ytr): lut.observe(row,int(label))
    tables[n]=lut

def eval_combo(names):
    preds=[]
    for i in range(len(yte)):
        scores=[0]*10
        hits=0
        for n in names:
            s,h=tables[n].score(test_feats[n][i])
            hits+=h
            for c in range(10): scores[c]+=s[c]
        order=sorted(range(10),key=lambda c:scores[c],reverse=True)
        preds.append(-1 if hits==0 or scores[order[0]]==scores[order[1]] else order[0])
    p=np.asarray(preds)
    cov=p>=0
    return {
        "accuracy_all":float(np.mean(p==yte)),
        "accuracy_covered":float(np.mean(p[cov]==yte[cov])) if cov.any() else 0.0,
        "coverage":float(np.mean(cov))
    }

combos={
    "pixel_control":["pixel_control"],
    "adi9":["adi9"],
    "walk7":["walk7"],
    "vof_d2":["vof_d2"],
    "s5":["s5"],
    "adi9_walk7":["adi9","walk7"],
    "adi9_vof":["adi9","vof_d2"],
    "adi9_s5":["adi9","s5"],
    "adi9_walk7_vof":["adi9","walk7","vof_d2"],
    "adi9_walk7_vof_s5":["adi9","walk7","vof_d2","s5"],
}
results={n:eval_combo(parts) for n,parts in combos.items()}
out={
    "dataset":{"samples":len(y),"train":len(train_idx),"test":len(test_idx)},
    "LoG_sigma":0.8,
    "scale":scale,
    "feature_widths":{n:len(train_feats[n][0]) for n in families},
    "results":results,
    "best":max(results,key=lambda n:results[n]["accuracy_all"])
}
print(json.dumps(out,indent=2))
(ROOT/"results/vit_vof_ablation_reproduced.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
