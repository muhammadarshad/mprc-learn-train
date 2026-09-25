"""v5 — Ring-native MEASURE classifier with directional event learning.

Remove the remaining dense linear logit layer.

No:
- W dot x logits
- bias logits
- softmax
- floating learning rate
- gradient magnitude
- backward matrix credit
- activation derivative

State
-----
Ten learned class states P_c in Z256^64.

Inference
---------
    E_c(x) = sum_f cdist(P_c[f], x[f])
    SELECT = argmin_c E_c

Training event
--------------
For a target y and closest rival r, if the margin is violated:
- move P_y one ring direction TOWARD x
- move P_r one ring direction AWAY from x

The requested +/-1 movements are not immediately applied. They are stored in
four directional/polarity residual memories (+U,+D,-U,-D) and emit only after
DEN events.

Three denominators (64,256,1000) test the event-time scale. DEN=1000 directly
corresponds to counting 1000 unit observations before one state quantum moves.

Prototype initialization is the exact training-class ring medoid; no test labels
or test states participate.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits

SEED=20260925
EPOCHS=120
MARGIN=64
NCLASS=10
FDIM=64


def cdist_vec(a,b):
    aa=np.asarray(a,dtype=np.int16)
    bb=np.asarray(b,dtype=np.int16)
    ab=(aa-bb)&255
    ba=(bb-aa)&255
    return np.minimum(ab,ba)


def energies(P,x):
    return cdist_vec(P,x[None,:]).sum(axis=1,dtype=np.int64)


def ring_toward(cur,target):
    """One signed shortest-step direction on Z256: +1, -1, or 0.

    Antipodal distance 128 is genuinely ambiguous; emit no direction and count it.
    """
    c=np.asarray(cur,dtype=np.int16)
    t=np.asarray(target,dtype=np.int16)
    cw=(t-c)&255
    ccw=(c-t)&255
    return np.where(cw<ccw,1,np.where(ccw<cw,-1,0)).astype(np.int64)


def polarity(v):
    """MPRC origin 128: lower side positive, upper side negative."""
    x=np.asarray(v,dtype=np.int64)&255
    return np.where(x<=128,1,-1)


class D4RingStates:
    def __init__(self,value,den):
        self.value=np.asarray(value,dtype=np.int64).copy()&255
        self.den=int(den)
        self.residual=np.zeros(self.value.shape+(4,),dtype=np.int64)
        self.emitted=np.zeros(4,dtype=np.int64)

    def step_delta(self,req,mask_rows):
        """req is [K,F] signed direction; only selected class rows are active."""
        req=np.asarray(req,dtype=np.int64)
        active=np.zeros_like(self.value,dtype=bool)
        for r in mask_rows:
            active[int(r),:]=True
        req=np.where(active,req,0)

        mag=np.abs(req)
        pol=polarity(self.value)
        up=req>0
        down=req<0
        plus=pol>0
        minus=pol<0
        masks=(plus&up,plus&down,minus&up,minus&down)
        delta=np.zeros_like(self.value)

        for d,m in enumerate(masks):
            if not np.any(m):
                continue
            rr=self.residual[...,d]
            rr[m]+=mag[m]
            k=np.zeros_like(rr)
            k[m]=rr[m]//self.den
            rr[m]-=k[m]*self.den
            self.emitted[d]+=int(k.sum())
            if d in (0,2):
                delta+=k
            else:
                delta-=k

        self.value=(self.value+delta)&255
        assert np.all((self.residual>=0)&(self.residual<self.den))


def exact_class_medoids(X,y):
    """Choose one observed state/class minimizing total within-class ring distance."""
    P=np.empty((NCLASS,FDIM),dtype=np.uint8)
    ids=[]
    for c in range(NCLASS):
        I=np.where(y==c)[0]
        Z=X[I]
        # [m,m,64], small for sklearn digits; exact integer ring distance.
        a=Z[:,None,:].astype(np.int16)
        b=Z[None,:,:].astype(np.int16)
        ab=(a-b)&255
        ba=(b-a)&255
        D=np.minimum(ab,ba).sum(axis=2,dtype=np.int64)
        sums=D.sum(axis=1,dtype=np.int64)
        j=int(np.argmin(sums))
        P[c]=Z[j]
        ids.append(int(I[j]))
    return P,ids


def evaluate(P,X,y):
    correct=0
    for x,yy in zip(X,y):
        e=energies(P,x)
        pred=int(np.argmin(e))
        correct+=int(pred==int(yy))
    return correct,len(y)


def train(P0,Xtr,ytr,Xte,yte,den):
    S=D4RingStates(P0,den)
    rng=np.random.default_rng(SEED+den)
    hist=[]
    best=(0,0)
    antipodal=0
    events=0

    for epoch in range(EPOCHS):
        order=rng.permutation(len(Xtr))
        violations=0

        for ii in order:
            x=Xtr[ii]
            yy=int(ytr[ii])
            e=energies(S.value,x)

            rivals=e.copy()
            rivals[yy]=np.iinfo(np.int64).max
            r=int(np.argmin(rivals))

            if int(e[yy])+MARGIN <= int(e[r]):
                continue

            violations+=1
            events+=1

            dy=ring_toward(S.value[yy],x)
            dr=-ring_toward(S.value[r],x)
            antipodal+=int(np.count_nonzero(dy==0))+int(np.count_nonzero(dr==0))

            req=np.zeros_like(S.value,dtype=np.int64)
            req[yy]=dy
            req[r]=dr
            S.step_delta(req,[yy,r])

        tr=evaluate(S.value,Xtr,ytr)
        te=evaluate(S.value,Xte,yte)
        if te[0]>best[0]:
            best=(te[0],epoch+1)
        hist.append({
            "epoch":epoch+1,
            "violations":violations,
            "train_correct":tr[0],"train_total":tr[1],
            "test_correct":te[0],"test_total":te[1],
        })
        print("DEN",den,hist[-1],flush=True)

    return {
        "final_train":list(evaluate(S.value,Xtr,ytr)),
        "final_test":list(evaluate(S.value,Xte,yte)),
        "best_test":[best[0],len(yte)],
        "best_epoch":best[1],
        "emitted":S.emitted.tolist(),
        "total_violation_events":events,
        "zero_direction_events_including_equal_or_antipodal":antipodal,
        "history":hist,
    }


def ring_direction_gate():
    tested=0
    ties=0
    for a in range(256):
        for b in range(256):
            d=int(ring_toward(np.asarray([a]),np.asarray([b]))[0])
            before=int(cdist_vec(np.asarray([a]),np.asarray([b]))[0])
            after=int(cdist_vec(np.asarray([(a+d)&255]),np.asarray([b]))[0])
            if before==0 or before==128:
                assert d==0
                if before==128: ties+=1
            else:
                assert after==before-1
            tested+=1
    assert ties==256
    return {"pass":True,"ordered_pairs":tested,"antipodal_ties":ties}


gate=ring_direction_gate()

ds=load_digits()
X=(np.asarray(ds.data,dtype=np.int64)*15).astype(np.uint8)
y=np.asarray(ds.target,dtype=np.int64)
rng=np.random.default_rng(SEED)
order=rng.permutation(len(X))
cut=int(len(order)*.7)
trix=order[:cut];teix=order[cut:]
Xtr=X[trix];ytr=y[trix]
Xte=X[teix];yte=y[teix]

P0,medoid_ids=exact_class_medoids(Xtr,ytr)
static_train=evaluate(P0,Xtr,ytr)
static_test=evaluate(P0,Xte,yte)

results={}
for den in (64,256,1000):
    results[str(den)]=train(P0,Xtr,ytr,Xte,yte,den)
    print("RESULT",den,results[str(den)]["final_test"],"best",results[str(den)]["best_test"],flush=True)

report={
    "model":"v5-ring-measure-directional-prototypes",
    "ring_direction_gate":gate,
    "state":"10 class prototypes x 64 Z256 states",
    "inference":"SELECT argmin_c sum_f cdist(P_c[f],x[f])",
    "linear_logits":False,
    "softmax":False,
    "float_learning_rate":False,
    "gradient":False,
    "backward_matrix_credit":False,
    "activation_derivative":False,
    "initialization":{
        "type":"exact training-class ring medoid",
        "medoid_training_indices":medoid_ids,
        "static_train":list(static_train),
        "static_test":list(static_test),
    },
    "training":{
        "margin":MARGIN,
        "epochs":EPOCHS,
        "event":"target prototype toward sample; closest rival away",
        "memory_bins":["+U","+D","-U","-D"],
        "denominators":[64,256,1000],
    },
    "results":results,
    "claim_boundary":(
        "This probe removes dense linear logits and gradient-based credit entirely. "
        "It is a supervised prototype learner because class labels choose the target "
        "prototype. Success supports a ring-native MEASURE/SELECT learning route but "
        "does not establish it as the final MPRC network or attention mechanism."
    )
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"ring_measure_directional_prototypes_v5.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
