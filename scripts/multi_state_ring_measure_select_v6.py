"""v6 — Multi-state ring MEASURE/SELECT memory.

v5 removed linear logits, softmax, gradient magnitude, and backward credit but
compressed each class into ONE Z256^64 prototype. Best test accuracy was limited.

v6 asks whether that loss is representational rather than a failure of
ring-native learning.

Each class owns K learned states:
    P[c,k] in Z256^64

Inference:
    E[c,k](x) = sum_f cdist(P[c,k,f], x[f])
    SELECT = class of globally minimum energy

Training:
    target = nearest state within the true class
    rival  = nearest state among all wrong classes
    if target margin is violated:
        target state moves toward x
        rival state moves away from x
    movements are accumulated in (+U,+D,-U,-D) D4 memory.

No GEMV/logit layer, no softmax, no float learning rate, no gradient, no
activation derivative, no backward matrix credit.

Initialization:
    deterministic integer farthest-first states within each training class.
    First state is the exact class medoid; further states maximize distance to
    the already selected class states.

K in {1,2,4,7}; denominator fixed to 64 because v5 established it as the
strongest of {64,256,1000}. K=7 is tested, not assumed superior.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits

SEED=20260925
EPOCHS=100
MARGIN=64
NCLASS=10
FDIM=64
DEN=64
KS=(1,2,4,7)


def cdist(a,b):
    aa=np.asarray(a,dtype=np.int16)
    bb=np.asarray(b,dtype=np.int16)
    ab=(aa-bb)&255
    ba=(bb-aa)&255
    return np.minimum(ab,ba)


def energies(P,x):
    # P [C,K,F] -> [C,K]
    return cdist(P,x[None,None,:]).sum(axis=2,dtype=np.int64)


def ring_toward(cur,target):
    c=np.asarray(cur,dtype=np.int16)
    t=np.asarray(target,dtype=np.int16)
    cw=(t-c)&255
    ccw=(c-t)&255
    return np.where(cw<ccw,1,np.where(ccw<cw,-1,0)).astype(np.int64)


def polarity(v):
    x=np.asarray(v,dtype=np.int64)&255
    return np.where(x<=128,1,-1)


class D4Memory:
    def __init__(self,value,den=DEN):
        self.value=np.asarray(value,dtype=np.int64).copy()&255
        self.den=int(den)
        self.residual=np.zeros(self.value.shape+(4,),dtype=np.int64)
        self.emitted=np.zeros(4,dtype=np.int64)

    def step_pair(self,c1,k1,req1,c2,k2,req2):
        req=np.zeros_like(self.value,dtype=np.int64)
        active=np.zeros_like(self.value,dtype=bool)
        req[c1,k1]=req1
        req[c2,k2]=req2
        active[c1,k1]=True
        active[c2,k2]=True

        mag=np.abs(req)
        pol=polarity(self.value)
        up=req>0
        down=req<0
        plus=pol>0
        minus=pol<0
        masks=(plus&up,plus&down,minus&up,minus&down)
        delta=np.zeros_like(self.value)

        for d,m in enumerate(masks):
            m &= active
            if not np.any(m):
                continue
            r=self.residual[...,d]
            r[m]+=mag[m]
            k=np.zeros_like(r)
            k[m]=r[m]//self.den
            r[m]-=k[m]*self.den
            self.emitted[d]+=int(k.sum())
            if d in (0,2):
                delta+=k
            else:
                delta-=k

        self.value=(self.value+delta)&255
        assert np.all((self.residual>=0)&(self.residual<self.den))


def class_distance_matrix(Z):
    a=Z[:,None,:].astype(np.int16)
    b=Z[None,:,:].astype(np.int16)
    ab=(a-b)&255
    ba=(b-a)&255
    return np.minimum(ab,ba).sum(axis=2,dtype=np.int64)


def farthest_first_states(X,y,K):
    P=np.empty((NCLASS,K,FDIM),dtype=np.uint8)
    ids=np.empty((NCLASS,K),dtype=np.int64)

    for c in range(NCLASS):
        I=np.where(y==c)[0]
        Z=X[I]
        D=class_distance_matrix(Z)

        # First state = exact class medoid.
        med=int(np.argmin(D.sum(axis=1,dtype=np.int64)))
        chosen=[med]

        while len(chosen)<K:
            # Distance of each candidate to its nearest already selected state.
            near=D[:,chosen].min(axis=1)
            near[np.asarray(chosen,dtype=np.int64)]=-1
            # deterministic tie break from argmax's first occurrence
            nxt=int(np.argmax(near))
            chosen.append(nxt)

        for k,j in enumerate(chosen):
            P[c,k]=Z[j]
            ids[c,k]=int(I[j])

    return P,ids


def evaluate(P,X,y):
    ok=0
    for x,yy in zip(X,y):
        E=energies(P,x)
        c,k=np.unravel_index(int(np.argmin(E)),E.shape)
        ok+=int(c==int(yy))
    return ok,len(y)


def train(P0,Xtr,ytr,Xte,yte,K):
    S=D4Memory(P0)
    rng=np.random.default_rng(SEED+K)
    hist=[]
    best=(0,0)
    events=0

    for epoch in range(EPOCHS):
        order=rng.permutation(len(Xtr))
        vio=0

        for ii in order:
            x=Xtr[ii]
            yy=int(ytr[ii])
            E=energies(S.value,x)

            # nearest target state in correct class
            kt=int(np.argmin(E[yy]))
            et=int(E[yy,kt])

            # globally nearest wrong-class state
            Ew=E.copy()
            Ew[yy,:]=np.iinfo(np.int64).max
            cr,kr=np.unravel_index(int(np.argmin(Ew)),Ew.shape)
            er=int(Ew[cr,kr])

            if et+MARGIN<=er:
                continue

            vio+=1
            events+=1
            toward=ring_toward(S.value[yy,kt],x)
            away=-ring_toward(S.value[cr,kr],x)
            S.step_pair(yy,kt,toward,cr,kr,away)

        tr=evaluate(S.value,Xtr,ytr)
        te=evaluate(S.value,Xte,yte)
        if te[0]>best[0]:
            best=(te[0],epoch+1)

        hist.append({
            "epoch":epoch+1,
            "violations":vio,
            "train_correct":tr[0],"train_total":tr[1],
            "test_correct":te[0],"test_total":te[1],
        })
        print("K",K,hist[-1],flush=True)

    return {
        "final_train":list(evaluate(S.value,Xtr,ytr)),
        "final_test":list(evaluate(S.value,Xte,yte)),
        "best_test":[best[0],len(yte)],
        "best_epoch":best[1],
        "events":events,
        "emitted":S.emitted.tolist(),
        "history":hist,
    }


def selection_gate():
    # Exhaustive local distance metric symmetry / zero / adjacency.
    for a in range(256):
        assert int(cdist(np.asarray([a]),np.asarray([a]))[0])==0
        assert int(cdist(np.asarray([a]),np.asarray([(a+1)&255]))[0])==1
        assert int(cdist(np.asarray([a]),np.asarray([(a-1)&255]))[0])==1

    # SELECT must choose exact state when one is present.
    P=np.zeros((2,2,FDIM),dtype=np.uint8)
    P[0,0,:]=17
    P[0,1,:]=80
    P[1,0,:]=200
    P[1,1,:]=250
    x=np.full(FDIM,200,dtype=np.uint8)
    E=energies(P,x)
    c,k=np.unravel_index(int(np.argmin(E)),E.shape)
    assert (c,k)==(1,0)
    assert int(E[1,0])==0
    return {"pass":True}


gate=selection_gate()

ds=load_digits()
X=(np.asarray(ds.data,dtype=np.int64)*15).astype(np.uint8)
y=np.asarray(ds.target,dtype=np.int64)
rng=np.random.default_rng(SEED)
order=rng.permutation(len(X))
cut=int(len(order)*.7)
trix=order[:cut];teix=order[cut:]
Xtr=X[trix];ytr=y[trix]
Xte=X[teix];yte=y[teix]

results={}
for K in KS:
    P0,ids=farthest_first_states(Xtr,ytr,K)
    static_train=evaluate(P0,Xtr,ytr)
    static_test=evaluate(P0,Xte,yte)
    learned=train(P0,Xtr,ytr,Xte,yte,K)

    results[str(K)]={
        "state_count":NCLASS*K,
        "initialization_ids":ids.tolist(),
        "static_train":list(static_train),
        "static_test":list(static_test),
        "learned":learned,
    }
    print(
        "RESULT K",K,
        "static",static_test,
        "final",learned["final_test"],
        "best",learned["best_test"],
        "epoch",learned["best_epoch"],
        flush=True,
    )

report={
    "model":"v6-multi-state-ring-measure-select",
    "gate":gate,
    "K_values":list(KS),
    "denominator":DEN,
    "epochs":EPOCHS,
    "margin":MARGIN,
    "state":"P[c,k] in Z256^64",
    "initialization":"class medoid + deterministic integer farthest-first states",
    "inference":"SELECT class of argmin_(c,k) sum_f cdist(P[c,k,f],x[f])",
    "training":"nearest true-class state toward x; nearest wrong-class state away; D4 event memory",
    "linear_logits":False,
    "softmax":False,
    "float_learning_rate":False,
    "gradient":False,
    "backward_matrix_credit":False,
    "activation_derivative":False,
    "results":results,
    "claim_boundary":(
        "v6 tests whether multiple ring states recover class geometry lost by v5's "
        "single-prototype compression. K=7 is included as an empirical candidate, "
        "not assumed optimal or promoted to theorem."
    )
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"multi_state_ring_measure_select_v6.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
