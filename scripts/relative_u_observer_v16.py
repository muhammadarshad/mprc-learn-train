"""v16 — Relative U-Observer: integrate movement over time.

v15 result:
    absolute displacement recovery : 77.54%
    velocity recovery              : 97.99%
    acceleration recovery          : 96.74%

That strongly suggests temporal RELATIVE motion is more observable than repeatedly
solving absolute position from global memory.

v16 therefore removes global-memory lookup after frame 0.

Trajectory:
    q_t = roll(x_0, d_t)

Observer state:
    D_0 = 0                         # inverse displacement anchor
    V_t = estimate(q_(t-1), q_t)    # inverse relative movement
    D_t = D_(t-1) + V_t
    A_t = V_t - V_(t-1)

For pairwise estimation, search v in [-4,+4]:
    roll(q_(t-1), -v) ~= q_t

Since q_t = roll(q_(t-1), delta_t), the exact observer velocity is:
    V_t = -delta_t

Canonicalization:
    q_canon_t = roll(q_t, D_t)

No label, class model, or TRAIN memory participates in movement estimation.

The clean classifier remains trained only on clean TRAIN data.

This is a controlled temporal mechanics probe. It is not yet the final physical
MPRC U law or INFORMATION codec.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit

DEN=1000
SPLIT_SEED=20260925
MODEL_SEEDS=(7,19,31,43,59)
EPOCHS=60
BATCH=64
MARGIN=16
NCLASS=10
FDIM=64
T=5
VELOCITY_RADIUS=4

TRAJECTORIES=(
    ("CV_-2", tuple(-2*t for t in range(T))),
    ("CV_-1", tuple(-1*t for t in range(T))),
    ("CV_+1", tuple( 1*t for t in range(T))),
    ("CV_+2", tuple( 2*t for t in range(T))),
    ("ACC_1_1",   tuple( 1*t + 1*t*(t-1)//2 for t in range(T))),
    ("ACC_2_-1",  tuple( 2*t - 1*t*(t-1)//2 for t in range(T))),
    ("ACC_-1_-1", tuple(-1*t - 1*t*(t-1)//2 for t in range(T))),
    ("ACC_-2_1",  tuple(-2*t + 1*t*(t-1)//2 for t in range(T))),
)

for name,d in TRAJECTORIES:
    dv=np.diff(np.asarray(d,dtype=np.int64))
    assert int(np.abs(dv).max())<=VELOCITY_RADIUS


def trunc_div_array(a,q):
    a=np.asarray(a,dtype=np.int64)
    out=np.empty_like(a)
    pos=a>=0
    out[pos]=a[pos]//q
    out[~pos]=-((-a[~pos])//q)
    return out


class ScalarResidualParam:
    def __init__(self,value):
        self.value=np.asarray(value,dtype=np.int64).copy()
        self.residual=np.zeros_like(self.value,dtype=np.int64)

    def step(self,grad):
        self.residual+=np.asarray(grad,dtype=np.int64)
        k=trunc_div_array(self.residual,DEN)
        self.residual-=k*DEN
        self.value-=k
        assert np.all(np.abs(self.residual)<DEN)


class LocalNet:
    def __init__(self,W0,b0,L0):
        self.L=ScalarResidualParam(L0)
        self.W=ScalarResidualParam(W0)
        self.b=ScalarResidualParam(b0)

    def features(self,X):
        return 128-(self.L.value[X]&255)

    def scores(self,X):
        a=self.features(X)
        return a,a@self.W.value+self.b.value

    def predict(self,X):
        return self.scores(X)[1].argmax(axis=1).astype(np.int64)

    def ckpt(self):
        return {"L":self.L.value.copy(),"W":self.W.value.copy(),"b":self.b.value.copy()}

    def restore(self,c):
        self.L.value=c["L"].copy()
        self.W.value=c["W"].copy()
        self.b.value=c["b"].copy()

    def train_select_validation(self,Xtr,ytr,Xva,yva,orders):
        best=(-1,0);best_ck=None
        for ep,order in enumerate(orders,1):
            for st in range(0,len(order),BATCH):
                ii=order[st:st+BATCH]
                x=Xtr[ii];yy=ytr[ii]
                a,s=self.scores(x)
                ds=np.zeros_like(s,dtype=np.int64)

                for i in range(len(ii)):
                    yi=int(yy[i]);sy=int(s[i,yi])
                    for c in range(NCLASS):
                        if c==yi:
                            continue
                        if int(s[i,c])+MARGIN>sy:
                            ds[i,c]+=1
                            ds[i,yi]-=1

                gW=a.T@ds
                gb=ds.sum(axis=0,dtype=np.int64)
                ga=ds@self.W.value.T
                gL=np.zeros(256,dtype=np.int64)
                np.add.at(gL,x.reshape(-1),(-ga).reshape(-1))

                self.W.step(gW);self.b.step(gb);self.L.step(gL)
                self.W.value=np.clip(self.W.value,-127,127)
                self.b.value=np.clip(self.b.value,-32767,32767)
                self.L.value&=255

            p=self.predict(Xva)
            ac=int((p==yva).sum())
            if ac>best[0]:
                best=(ac,ep)
                best_ck=self.ckpt()

        self.restore(best_ck)
        return {"selected_validation":[best[0],len(yva)],"selected_epoch":best[1]}


def cdist(a,b):
    aa=np.asarray(a,dtype=np.int16)
    bb=np.asarray(b,dtype=np.int16)
    ab=(aa-bb)&255
    ba=(bb-aa)&255
    return np.minimum(ab,ba)


def estimate_relative_inverse_velocity(prev,curr):
    """Estimate v such that roll(prev,-v) best matches curr."""
    best=None

    for v in range(-VELOCITY_RADIUS,VELOCITY_RADIUS+1):
        moved=np.roll(prev,-int(v))
        d=cdist(moved,curr)
        support=int((moved==curr).sum())
        energy=int(d.sum(dtype=np.int64))

        # More exact matches, lower ring energy, smaller motion magnitude,
        # then deterministic signed direction.
        key=(-support,energy,abs(v),v)

        if best is None or key<best[0]:
            best=(key,int(v),support,energy)

    _,v,s,e=best
    return v,s,e


def observe_sequence(frames):
    """Return integrated inverse displacement, relative velocity, acceleration."""
    F=np.asarray(frames,dtype=np.uint8)
    assert F.shape==(T,FDIM)

    D=np.zeros(T,dtype=np.int64)
    V=np.zeros(T-1,dtype=np.int64)
    support=np.empty(T-1,dtype=np.int64)
    energy=np.empty(T-1,dtype=np.int64)

    for t in range(1,T):
        v,s,e=estimate_relative_inverse_velocity(F[t-1],F[t])
        V[t-1]=v
        support[t-1]=s
        energy[t-1]=e
        D[t]=D[t-1]+v

    A=np.diff(V)

    C=np.empty_like(F)
    for t in range(T):
        C[t]=np.roll(F[t],int(D[t]))

    return D,V,A,C,support,energy


def temporal_gate():
    rng=np.random.default_rng(16)
    trials=128

    for _ in range(trials):
        x=rng.integers(0,256,size=FDIM,dtype=np.uint8)

        for name,dseq in TRAJECTORIES:
            F=np.asarray([np.roll(x,int(d)) for d in dseq],dtype=np.uint8)
            D,V,A,C,S,E=observe_sequence(F)

            true_D=-np.asarray(dseq,dtype=np.int64)
            true_V=np.diff(true_D)
            true_A=np.diff(true_V)

            assert np.array_equal(D,true_D)
            assert np.array_equal(V,true_V)
            assert np.array_equal(A,true_A)
            assert np.all(C==x[None,:])
            assert np.all(E==0)
            assert np.all(S==FDIM)

    return {"pass":True,"random_clean_states":trials,"trajectory_count":len(TRAJECTORIES)}


gate=temporal_gate()

# Fixed clean split.
ds=load_digits()
X=(np.asarray(ds.data,dtype=np.int64)*15).astype(np.uint8)
y=np.asarray(ds.target,dtype=np.int64)

outer=StratifiedShuffleSplit(n_splits=1,test_size=.20,random_state=SPLIT_SEED)
tv,te=next(outer.split(X,y))
inner=StratifiedShuffleSplit(n_splits=1,test_size=.25,random_state=SPLIT_SEED+1)
trr,var=next(inner.split(X[tv],y[tv]))
tr=tv[trr];va=tv[var]

Xtr=X[tr];ytr=y[tr]
Xva=X[va];yva=y[va]
Xte=X[te];yte=y[te]

# Build temporal test once; observer is model-independent.
J=len(TRAJECTORIES)
S=len(Xte)

frames=np.empty((S,J,T,FDIM),dtype=np.uint8)
true_D=np.empty((S,J,T),dtype=np.int64)
obs_D=np.empty((S,J,T),dtype=np.int64)
obs_V=np.empty((S,J,T-1),dtype=np.int64)
obs_A=np.empty((S,J,T-2),dtype=np.int64)
canon=np.empty_like(frames)
support=np.empty((S,J,T-1),dtype=np.int64)
energy=np.empty((S,J,T-1),dtype=np.int64)

for s,x in enumerate(Xte):
    for j,(name,dseq) in enumerate(TRAJECTORIES):
        F=np.asarray([np.roll(x,int(d)) for d in dseq],dtype=np.uint8)
        D,V,A,C,Sup,E=observe_sequence(F)

        frames[s,j]=F
        canon[s,j]=C
        obs_D[s,j]=D
        obs_V[s,j]=V
        obs_A[s,j]=A
        support[s,j]=Sup
        energy[s,j]=E
        true_D[s,j]=-np.asarray(dseq,dtype=np.int64)

true_V=np.diff(true_D,axis=2)
true_A=np.diff(true_V,axis=2)

disp_correct=int((obs_D==true_D).sum())
vel_correct=int((obs_V==true_V).sum())
acc_correct=int((obs_A==true_A).sum())

path_exact=int(np.all(obs_D==true_D,axis=2).sum())

flat_frames=frames.reshape(-1,FDIM)
flat_canon=canon.reshape(-1,FDIM)
flat_labels=np.repeat(yte,J*T)

per_seed=[]

for seed in MODEL_SEEDS:
    rng=np.random.default_rng(seed)
    W0=rng.integers(-2,3,size=(FDIM,NCLASS),dtype=np.int64)
    b0=np.zeros(NCLASS,dtype=np.int64)
    L0=np.arange(256,dtype=np.int64)

    rr=np.random.default_rng(seed+100000)
    orders=[rr.permutation(len(Xtr)) for _ in range(EPOCHS)]

    net=LocalNet(W0,b0,L0)
    sel=net.train_select_validation(Xtr,ytr,Xva,yva,orders)

    clean_correct=int((net.predict(Xte)==yte).sum())
    raw_correct=int((net.predict(flat_frames)==flat_labels).sum())
    canon_correct=int((net.predict(flat_canon)==flat_labels).sum())

    per_seed.append({
        "seed":seed,
        "local_selection":sel,
        "clean_correct":clean_correct,
        "clean_total":len(yte),
        "raw_temporal_correct":raw_correct,
        "canonicalized_temporal_correct":canon_correct,
        "temporal_total":len(flat_labels),
        "delta_correct":canon_correct-raw_correct,
    })

    print(
        "SEED",seed,
        "CLEAN",clean_correct,"/",len(yte),
        "RAW",raw_correct,"/",len(flat_labels),
        "CANON",canon_correct,"/",len(flat_labels),
        "DELTA",canon_correct-raw_correct,
        flush=True,
    )


trajectory_rows=[]
for j,(name,dseq) in enumerate(TRAJECTORIES):
    trajectory_rows.append({
        "name":name,
        "true_displacement":list(dseq),
        "true_inverse_state":true_D[0,j].tolist(),
        "true_velocity":true_V[0,j].tolist(),
        "true_acceleration":true_A[0,j].tolist(),
        "displacement_correct":int((obs_D[:,j]==true_D[:,j]).sum()),
        "displacement_total":S*T,
        "velocity_correct":int((obs_V[:,j]==true_V[:,j]).sum()),
        "velocity_total":S*(T-1),
        "acceleration_correct":int((obs_A[:,j]==true_A[:,j]).sum()),
        "acceleration_total":S*(T-2),
        "exact_path_count":int(np.all(obs_D[:,j]==true_D[:,j],axis=1).sum()),
        "path_total":S,
    })


raw_sum=sum(r["raw_temporal_correct"] for r in per_seed)
canon_sum=sum(r["canonicalized_temporal_correct"] for r in per_seed)

report={
    "model":"v16-relative-U-observer",
    "parent_benchmark":{
        "v15_absolute_displacement":"11166/14400",
        "v15_velocity":"11288/11520",
        "v15_acceleration":"8358/8640",
        "v15_canonicalized_classification":"62871/72000",
    },
    "gate":gate,
    "observer":{
        "anchor":"D_0=0",
        "velocity_estimation":"pairwise q_(t-1) -> q_t only",
        "velocity_radius":VELOCITY_RADIUS,
        "displacement_integration":"D_t=D_(t-1)+V_t",
        "acceleration":"A_t=V_t-V_(t-1)",
        "train_memory_used_after_anchor":False,
        "labels_used":False,
    },
    "recovery":{
        "displacement_correct":disp_correct,
        "displacement_total":true_D.size,
        "velocity_correct":vel_correct,
        "velocity_total":true_V.size,
        "acceleration_correct":acc_correct,
        "acceleration_total":true_A.size,
        "exact_path_count":path_exact,
        "path_total":S*J,
        "support_mean":float(support.mean()),
        "energy_mean":float(energy.mean()),
    },
    "trajectory_breakdown":trajectory_rows,
    "classification":{
        "per_seed":per_seed,
        "aggregate_raw_correct":raw_sum,
        "aggregate_canonicalized_correct":canon_sum,
        "aggregate_total":len(MODEL_SEEDS)*len(flat_labels),
        "delta_correct":canon_sum-raw_sum,
    },
    "claim_boundary":(
        "v16 demonstrates a relative temporal observer on synthetic GEN7-orbit motion. "
        "It does not yet define physical time units, final U rotation semantics, or the "
        "1,920-byte INFORMATION encoding."
    ),
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"relative_u_observer_v16.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
