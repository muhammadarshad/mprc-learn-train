"""Pure-integer training v0: no float LR, no softmax, trainable byte LUT.

Purpose
-------
Test three loose ends in one real-data training loop:

1. Gradient step representation:
       eta = 1/1000
   implemented by exact residual integer accumulators.

2. Softmax:
   absent. Training uses an integer multiclass margin objective; inference uses
   score ordering/argmax only.

3. SiLU:
   absent. A shared 256-entry integer activation LUT is trainable directly from
   downstream error. This is a compatibility experiment for learned REACT/LUT
   activation, not a claim that the learned table is already the final MPRC REACT law.

Attention is intentionally not re-invented here. The workflow runs the separately
gated Global Attention probe before this trainer.

All model state and training arithmetic below are integer NumPy arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits

Q=1000
SEED=20260925
EPOCHS=40
BATCH=64
MARGIN=16
HIDDEN_FEATURES=64
NCLASS=10


def trunc_div_array(a:np.ndarray,q:int)->np.ndarray:
    pos=a>=0
    out=np.empty_like(a,dtype=np.int64)
    out[pos]=a[pos]//q
    out[~pos]=-((-a[~pos])//q)
    return out


class ResidualParam:
    def __init__(self,value:np.ndarray,Q:int=Q):
        self.value=value.astype(np.int64,copy=True)
        self.residual=np.zeros_like(self.value,dtype=np.int64)
        self.Q=int(Q)
        self.positive_quanta=0
        self.negative_quanta=0

    def step(self,grad:np.ndarray):
        self.residual += grad.astype(np.int64,copy=False)
        k=trunc_div_array(self.residual,self.Q)
        self.residual -= k*self.Q
        # SGD: parameter -= emitted gradient quantum.
        self.value -= k
        self.positive_quanta += int(np.maximum(k,0).sum())
        self.negative_quanta += int(np.maximum(-k,0).sum())
        assert np.all(np.abs(self.residual) < self.Q)


# ------------------------------------------------------------------
# Data: sklearn digits is integer 0..16. Scale to byte domain exactly.
# ------------------------------------------------------------------
ds=load_digits()
X=np.asarray(ds.data,dtype=np.int64)          # [1797,64], values 0..16
y=np.asarray(ds.target,dtype=np.int64)
Xbyte=X*15                                   # 0..240, exact integer map

rng=np.random.default_rng(SEED)
order=rng.permutation(len(Xbyte))
cut=int(len(order)*0.7)
train_ix=order[:cut]
test_ix=order[cut:]
Xtr=Xbyte[train_ix]; ytr=y[train_ix]
Xte=Xbyte[test_ix]; yte=y[test_ix]

# ------------------------------------------------------------------
# Trainable activation LUT replacing fixed SiLU.
# Initial table is a centered, low-gain integer identity compatibility seed.
# It is model state and is trained from downstream error.
# ------------------------------------------------------------------
lut0=((np.arange(256,dtype=np.int64)-128)//16)
L=ResidualParam(lut0)

# Linear readout over 64 activated features.
W0=rng.integers(-2,3,size=(HIDDEN_FEATURES,NCLASS),dtype=np.int64)
b0=np.zeros(NCLASS,dtype=np.int64)
W=ResidualParam(W0)
b=ResidualParam(b0)


def activated(xb:np.ndarray)->np.ndarray:
    return L.value[xb]


def scores(xb:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    a=activated(xb)
    s=a@W.value + b.value
    return a,s


def evaluate(xb,yb):
    a,s=scores(xb)
    p=s.argmax(axis=1)
    correct=int((p==yb).sum())
    return correct,len(yb)


history=[]
for epoch in range(EPOCHS):
    ep_order=rng.permutation(len(Xtr))
    violations_total=0

    for st in range(0,len(ep_order),BATCH):
        ids=ep_order[st:st+BATCH]
        xb=Xtr[ids]
        yb=ytr[ids]

        a,s=scores(xb)
        B=len(xb)

        # Integer multiclass margin objective.
        dscores=np.zeros_like(s,dtype=np.int64)
        for i in range(B):
            yi=int(yb[i])
            sy=int(s[i,yi])
            v=0
            for c in range(NCLASS):
                if c==yi:
                    continue
                if int(s[i,c]) + MARGIN > sy:
                    dscores[i,c] += 1
                    v += 1
            dscores[i,yi] -= v
            violations_total += v

        # Exact integer gradients for linear readout.
        gW=a.T@dscores
        gb=dscores.sum(axis=0,dtype=np.int64)

        # Error reaching activation outputs, before W changes.
        ga=dscores@W.value.T

        # Shared LUT gradient: every occurrence of byte u contributes its downstream error.
        gL=np.zeros(256,dtype=np.int64)
        np.add.at(gL,xb.reshape(-1),ga.reshape(-1))

        W.step(gW)
        b.step(gb)
        L.step(gL)

        # Keep model state in bounded integer domains to avoid unbounded probe growth.
        # These are probe safety bounds, not claimed MPRC constants.
        W.value=np.clip(W.value,-127,127)
        b.value=np.clip(b.value,-32767,32767)
        L.value=np.clip(L.value,-127,127)

    tr=evaluate(Xtr,ytr)
    te=evaluate(Xte,yte)
    history.append({
        "epoch":epoch+1,
        "violations":int(violations_total),
        "train_correct":tr[0],
        "train_total":tr[1],
        "test_correct":te[0],
        "test_total":te[1],
    })
    print(history[-1],flush=True)

final_train=evaluate(Xtr,ytr)
final_test=evaluate(Xte,yte)

report={
    "model":"pure-integer-v0",
    "runtime_float_model_state":False,
    "learning_rate":{"numerator":1,"denominator":Q,"float_lr":False},
    "softmax":False,
    "training_objective":"integer multiclass margin",
    "activation":"trainable shared 256-entry integer LUT",
    "fixed_silu":False,
    "gradient_arithmetic":"integer",
    "attention":"separate gated Global Attention probe; not yet integrated into this trainer",
    "dataset":{
        "name":"sklearn digits",
        "train":len(Xtr),
        "test":len(Xte),
        "input_byte_map":"digit_value*15 => 0..240",
    },
    "final":{
        "train_correct":final_train[0],
        "train_total":final_train[1],
        "test_correct":final_test[0],
        "test_total":final_test[1],
    },
    "update_telemetry":{
        "W_positive_gradient_quanta":W.positive_quanta,
        "W_negative_gradient_quanta":W.negative_quanta,
        "b_positive_gradient_quanta":b.positive_quanta,
        "b_negative_gradient_quanta":b.negative_quanta,
        "LUT_positive_gradient_quanta":L.positive_quanta,
        "LUT_negative_gradient_quanta":L.negative_quanta,
        "W_residual_abs_max":int(np.abs(W.residual).max()),
        "b_residual_abs_max":int(np.abs(b.residual).max()),
        "LUT_residual_abs_max":int(np.abs(L.residual).max()),
    },
    "lut":{
        "min":int(L.value.min()),
        "max":int(L.value.max()),
        "changed_entries":int(np.count_nonzero(L.value!=lut0)),
        "visited_input_entries":int(len(np.unique(Xtr))),
    },
    "history":history,
    "claim_boundary":(
        "This probe tests whether a real network can learn using integer residual "
        "updates, a trainable integer activation LUT, and no softmax. It does not "
        "yet integrate Global Attention into the trainable graph and does not prove "
        "the final MPRC learning law."
    )
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"pure_integer_training_v0.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
