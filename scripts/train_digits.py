from __future__ import annotations
from pathlib import Path
import sys, json, time
import numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _data import load_vendored
from mprc_structural.features import log_batch, fit_z256_scale, to_z256, hv_edges_8x8, stride7_edges

X, y, train_idx, test_idx = load_vendored(ROOT)
tr_img, te_img = X[train_idx], X[test_idx]
ytr, yte = y[train_idx], y[test_idx]

log_tr = log_batch(tr_img, 0.8)
log_te = log_batch(te_img, 0.8)
scale = fit_z256_scale(log_tr)
Ztr, Zte = to_z256(log_tr, scale), to_z256(log_te, scale)
Ftr, Fte = Ztr.reshape(len(Ztr), -1), Zte.reshape(len(Zte), -1)

EDGES = hv_edges_8x8() + stride7_edges()
K = 10
W2 = [defaultdict(Counter) for _ in EDGES]
B1 = [defaultdict(Counter) for _ in EDGES]
S1 = [defaultdict(Counter) for _ in EDGES]

def delta(x,y): return (int(x)-int(y)) & 255
def sigma(x,y): return (int(x)+int(y)) & 255

t0=time.perf_counter()
for sample, cls in zip(Ftr, ytr):
    c=int(cls)
    for e,(i,j) in enumerate(EDGES):
        x,yv=int(sample[i]),int(sample[j])
        W2[e][(x,yv)][c]+=1
        B1[e][delta(x,yv)][c]+=1
        S1[e][sigma(x,yv)][c]+=1
train_seconds=time.perf_counter()-t0

def evidence(counter):
    if not counter: return None
    n=sum(counter.values())
    return [K*counter.get(c,0)-n for c in range(K)]

def predict(sample, use_sigma=False):
    scores=[0]*K
    hits=0
    for e,(i,j) in enumerate(EDGES):
        x,yv=int(sample[i]),int(sample[j])
        candidates=[W2[e].get((x,yv)), B1[e].get(delta(x,yv))]
        if use_sigma:
            candidates.append(S1[e].get(sigma(x,yv)))
        for counter in candidates:
            ev=evidence(counter)
            if ev is None: continue
            hits+=1
            for c in range(K): scores[c]+=ev[c]
    order=sorted(range(K), key=lambda c:scores[c], reverse=True)
    if hits==0 or scores[order[0]]==scores[order[1]]:
        return -1
    return order[0]

def eval_mprc(use_sigma=False):
    p=np.asarray([predict(s,use_sigma) for s in Fte])
    cov=p>=0
    return {
        "accuracy_all":float(np.mean(p==yte)),
        "accuracy_covered":float(np.mean(p[cov]==yte[cov])) if cov.any() else 0.0,
        "coverage":float(np.mean(cov))
    }

mprc_wb=eval_mprc(False)
mprc_all=eval_mprc(True)

logtr=log_tr.reshape(len(log_tr),-1)
logte=log_te.reshape(len(log_te),-1)
gd_float=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2500, solver="lbfgs", random_state=42))
gd_float.fit(logtr,ytr)
gd_float_acc=float(np.mean(gd_float.predict(logte)==yte))

gd_z=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2500, solver="lbfgs", random_state=42))
gd_z.fit(Ftr.astype(float),ytr)
gd_z_acc=float(np.mean(gd_z.predict(Fte.astype(float))==yte))

out={
    "dataset":{"samples":len(y),"train":len(train_idx),"test":len(test_idx)},
    "LoG_sigma":0.8,
    "scale":scale,
    "relations_per_image":len(EDGES),
    "mprc_W2_B1":mprc_wb,
    "mprc_W2_B1_Sigma":mprc_all,
    "gd_float_log":gd_float_acc,
    "gd_z256":gd_z_acc,
    "population_seconds":train_seconds,
    "W2_cells":sum(len(t) for t in W2),
    "B1_cells":sum(len(t) for t in B1),
    "Sigma_cells":sum(len(t) for t in S1)
}
print(json.dumps(out,indent=2))
(ROOT/"results/empirical_digits_reproduced.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
