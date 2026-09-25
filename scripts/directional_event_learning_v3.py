"""v3 — Event-count directional learning: remove gradient-magnitude descent.

Goal
====
Test the hypothesis that training can be expressed as accumulated discrete movement
events rather than a floating/continuous gradient-descent step.

There is no softmax and no activation derivative.

For every violated target-vs-rival margin:
- class error is a discrete event,
- feature polarity supplies movement direction,
- parameter polarity is recorded separately,
- movement events accumulate in (+U,+D,-U,-D),
- an update quantum is emitted only after DEN events.

Two update laws are compared on the same forward graph:
G-D4:
    magnitude-bearing integer gradient counts, accumulated directionally.
E-D4:
    event counts only. Feature magnitude is discarded from weight updates;
    LUT credit is reduced to sign only before counting.

This does NOT claim the event rule is already the final MPRC learning theorem.
It tests whether gradient magnitude is necessary for learning in this controlled net.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits

DEN=1000
SEED=20260925
EPOCHS=80
BATCH=64
MARGIN=16
NCLASS=10


def signed_coord(v,ring):
    x=np.asarray(v,dtype=np.int64)
    return (128-(x&255)) if ring else x


class D4Param:
    """Four residual bins: +U,+D,-U,-D.

    Input to step is an integer requested parameter delta (not an LR-scaled float).
    U/D records requested movement sign. +/- records current parameter polarity.
    """
    def __init__(self,value,ring=False):
        self.value=np.asarray(value,dtype=np.int64).copy()
        self.ring=bool(ring)
        self.residual=np.zeros(self.value.shape+(4,),dtype=np.int64)
        self.emitted=np.zeros(4,dtype=np.int64)

    def step_delta(self,delta_request):
        req=np.asarray(delta_request,dtype=np.int64)
        mag=np.abs(req)
        coord=signed_coord(self.value,self.ring)
        up=req>0
        down=req<0
        plus=coord>=0
        minus=~plus
        masks=(plus&up,plus&down,minus&up,minus&down)
        delta=np.zeros_like(self.value)

        for d,mask in enumerate(masks):
            if not np.any(mask):
                continue
            r=self.residual[...,d]
            r[mask]+=mag[mask]
            k=np.zeros_like(r)
            k[mask]=r[mask]//DEN
            r[mask]-=k[mask]*DEN
            self.emitted[d]+=int(k.sum())
            if d in (0,2):
                delta+=k
            else:
                delta-=k

        self.value+=delta
        if self.ring:
            self.value&=255
        assert np.all((self.residual>=0)&(self.residual<DEN))


def survival_gate():
    # Opposite unresolved motions survive independently.
    p=D4Param(np.asarray([10]))
    p.step_delta(np.asarray([999]))
    p.step_delta(np.asarray([-999]))
    assert int(p.value[0])==10
    assert int(p.residual[0,0])==999
    assert int(p.residual[0,1])==999

    # 1000th observation emits one motion quantum.
    p.step_delta(np.asarray([1]))
    assert int(p.value[0])==11
    p.step_delta(np.asarray([-1]))
    assert int(p.value[0])==10

    # Ring polarity splits same requested direction into different memory bins.
    a=D4Param(np.asarray([64]),ring=True)
    b=D4Param(np.asarray([192]),ring=True)
    a.step_delta(np.asarray([1000]))
    b.step_delta(np.asarray([1000]))
    assert int(a.emitted[0])==1
    assert int(b.emitted[2])==1

    return {"pass":True,"denominator":DEN,"bins":["+U","+D","-U","-D"]}


class Net:
    def __init__(self,feature_dim,seed):
        rng=np.random.default_rng(seed)
        self.L=D4Param(np.arange(256,dtype=np.int64),ring=True)
        self.W=D4Param(rng.integers(-2,3,size=(feature_dim,NCLASS),dtype=np.int64))
        self.b=D4Param(np.zeros(NCLASS,dtype=np.int64))
        self.L0=self.L.value.copy()

    def forward(self,idx):
        ring=self.L.value[idx]
        a=128-ring
        s=a@self.W.value+self.b.value
        return a,s

    def evaluate(self,idx,y):
        _,s=self.forward(idx)
        p=s.argmax(axis=1)
        return int((p==y).sum()),len(y)

    def train(self,itr,ytr,ite,yte,mode):
        """mode G = magnitude-bearing gradient counts; E = sign/event counts."""
        rng=np.random.default_rng(SEED+400+(0 if mode=="G" else 1))
        hist=[]
        best=(0,0)

        for epoch in range(EPOCHS):
            order=rng.permutation(len(itr))
            violations=0
            event_abs=0

            for st in range(0,len(order),BATCH):
                ids=order[st:st+BATCH]
                idx=itr[ids]
                yy=ytr[ids]
                a,s=self.forward(idx)

                ds=np.zeros_like(s,dtype=np.int64)
                for i in range(len(ids)):
                    yi=int(yy[i])
                    sy=int(s[i,yi])
                    v=0
                    for c in range(NCLASS):
                        if c==yi:
                            continue
                        if int(s[i,c])+MARGIN>sy:
                            ds[i,c]+=1
                            v+=1
                    ds[i,yi]-=v
                    violations+=v

                if mode=="G":
                    # Parameter delta is -gradient; all arithmetic remains integer.
                    dW=-(a.T@ds)
                    db=-ds.sum(axis=0,dtype=np.int64)

                    # a = 128-L[state]. Gradient-derived parameter delta for L.
                    credit=ds@self.W.value.T
                    dL_events=credit

                elif mode=="E":
                    # Pure event-count readout update:
                    # each violation contributes only the sign of feature activation.
                    sa=np.sign(a).astype(np.int64)
                    dW=-(sa.T@ds)
                    db=-ds.sum(axis=0,dtype=np.int64)

                    # Discrete target/rival credit. Keep only sign, never magnitude.
                    # No derivative of LUT or activation is evaluated.
                    raw_credit=ds@self.W.value.T
                    dL_events=np.sign(raw_credit).astype(np.int64)
                else:
                    raise KeyError(mode)

                # Shared LUT accumulates one signed event per observed state/feature.
                dL=np.zeros(256,dtype=np.int64)
                np.add.at(dL,idx.reshape(-1),dL_events.reshape(-1))

                event_abs += int(np.abs(dW).sum()+np.abs(db).sum()+np.abs(dL).sum())

                self.W.step_delta(dW)
                self.b.step_delta(db)
                self.L.step_delta(dL)

                # Bounds are execution safety for this probe, not MPRC constants.
                self.W.value=np.clip(self.W.value,-127,127)
                self.b.value=np.clip(self.b.value,-32767,32767)

            tr=self.evaluate(itr,ytr)
            te=self.evaluate(ite,yte)
            if te[0]>best[0]:
                best=(te[0],epoch+1)
            hist.append({
                "epoch":epoch+1,
                "violations":int(violations),
                "event_abs":int(event_abs),
                "train_correct":tr[0],"train_total":tr[1],
                "test_correct":te[0],"test_total":te[1],
            })
            print(mode,hist[-1],flush=True)

        return {
            "final_test":list(self.evaluate(ite,yte)),
            "best_test":[best[0],len(yte)],
            "best_epoch":best[1],
            "lut_changed":int(np.count_nonzero(self.L.value!=self.L0)),
            "emitted":{
                "LUT":self.L.emitted.tolist(),
                "W":self.W.emitted.tolist(),
                "b":self.b.emitted.tolist(),
            },
            "residual_totals":{
                "LUT":self.L.residual.sum(axis=tuple(range(self.L.residual.ndim-1))).tolist(),
                "W":self.W.residual.sum(axis=tuple(range(self.W.residual.ndim-1))).tolist(),
                "b":self.b.residual.sum(axis=tuple(range(self.b.residual.ndim-1))).tolist(),
            },
            "history":hist,
        }


gate=survival_gate()

ds=load_digits()
X=(np.asarray(ds.data,dtype=np.int64)*15).astype(np.uint8)
y=np.asarray(ds.target,dtype=np.int64)

rng=np.random.default_rng(SEED)
order=rng.permutation(len(X))
cut=int(len(order)*.7)
trix=order[:cut];teix=order[cut:]
Xtr=X[trix];ytr=y[trix];Xte=X[teix];yte=y[teix]

results={}
for mode in ("G","E"):
    # Identical initial parameter seed between modes.
    net=Net(Xtr.shape[1],SEED+77)
    res=net.train(Xtr,ytr,Xte,yte,mode)
    results[mode]=res
    print(mode,"final",res["final_test"],"best",res["best_test"],"at",res["best_epoch"],flush=True)

report={
    "model":"v3-directional-event-learning",
    "survival_gate":gate,
    "forward":"local 64-byte digits -> trainable Z256 LUT -> integer linear scores",
    "objective":"integer multiclass margin violations",
    "softmax":False,
    "float_learning_rate":False,
    "activation_derivative":False,
    "branches":{
        "G":"magnitude-bearing integer gradient counts into D4 accumulator",
        "E":"event-only: sign(feature) readout events + sign(target/rival credit) LUT events into D4 accumulator",
    },
    "results":results,
    "decision_rule":(
        "If E learns materially above chance, analytical gradient magnitude is not "
        "necessary for this network to learn. Comparison with G measures the cost of "
        "discarding magnitude while retaining direction, polarity and accumulated history."
    ),
    "claim_boundary":(
        "E is a discrete mistake-driven credit-assignment rule. It does not use a "
        "floating learning rate, softmax, or activation derivative, but its target/rival "
        "credit sign is still an error-propagation mechanism. Therefore success would "
        "support gradient-magnitude-free training, not yet prove a final MPRC learning theorem."
    )
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"directional_event_learning_v3.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
