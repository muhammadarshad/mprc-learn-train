"""v15 — U-Observer temporal movement benchmark on the GEN7 orbit.

v14 established that movement exists as recoverable state:
- moved observations collapse local accuracy
- GEN7 displacement recovery restores most of it

v15 promotes one-shot displacement d into a TIME SEQUENCE.

For a clean state x0, synthesize trajectories:
    x_t = roll(x0, d_t)

The observer estimates d_t independently from TRAIN memory, then derives:
    velocity     v_t = d_t - d_(t-1)
    acceleration a_t = v_t - v_(t-1)

All values are ordinary signed logical GEN7 steps in this controlled probe.
One logical step corresponds to one generator-7 physical-row step.

Trajectory families are predeclared:
A. constant velocity:
       d_t = v*t, v in {-2,-1,+1,+2}
B. accelerated:
       d_t = v0*t + a*t*(t-1)/2
       (v0,a) in {(1,1),(2,-1),(-1,-1),(-2,1)}

T = 5 frames (t=0..4). All displacements stay within search radius 14.

We measure:
1. frame displacement recovery
2. velocity recovery
3. acceleration recovery
4. frame classification before/after canonicalization
5. path identity: trajectories with same final displacement but different histories
   must remain distinguishable by velocity/acceleration sequence.

No temporal labels participate in displacement estimation.
The local classifier remains trained on CLEAN TRAIN only.
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
SEARCH_RADIUS=14

TRAJECTORIES=(
    ("CV_-2", tuple(-2*t for t in range(T))),
    ("CV_-1", tuple(-1*t for t in range(T))),
    ("CV_+1", tuple( 1*t for t in range(T))),
    ("CV_+2", tuple( 2*t for t in range(T))),
    ("ACC_1_1",  tuple( 1*t + 1*t*(t-1)//2 for t in range(T))),
    ("ACC_2_-1", tuple( 2*t - 1*t*(t-1)//2 for t in range(T))),
    ("ACC_-1_-1",tuple(-1*t - 1*t*(t-1)//2 for t in range(T))),
    ("ACC_-2_1", tuple(-2*t + 1*t*(t-1)//2 for t in range(T))),
)

for name,ds in TRAJECTORIES:
    assert len(ds)==T
    assert max(abs(int(x)) for x in ds)<=SEARCH_RADIUS


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
                x=Xtr[ii]; yy=ytr[ii]
                a,s=self.scores(x)
                ds=np.zeros_like(s,dtype=np.int64)
                for i in range(len(ii)):
                    yi=int(yy[i]); sy=int(s[i,yi])
                    for c in range(NCLASS):
                        if c==yi: continue
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

            pred=self.predict(Xva)
            ac=int((pred==yva).sum())
            if ac>best[0]:
                best=(ac,ep)
                best_ck=self.ckpt()

        self.restore(best_ck)
        return {"selected_validation":[best[0],len(yva)],"selected_epoch":best[1]}


def cdist_rows(q,X):
    qa=q.astype(np.int16)[None,:]
    xa=X.astype(np.int16)
    ab=(qa-xa)&255
    ba=(xa-qa)&255
    return np.minimum(ab,ba).sum(axis=1,dtype=np.int64)


class MovementEstimator:
    def __init__(self,Xmem):
        self.X=np.asarray(Xmem,dtype=np.uint8)
        self.N=len(self.X)

    def estimate_one(self,q,R=SEARCH_RADIUS):
        ids=np.arange(self.N,dtype=np.int64)
        best=None

        for d in range(-R,R+1):
            aligned=np.roll(self.X,-int(d),axis=1)
            sup=(aligned==q[None,:]).sum(axis=1,dtype=np.int64)
            E=cdist_rows(q,aligned)
            n=int(np.lexsort((ids,E,-sup))[0])
            key=(-int(sup[n]),int(E[n]),abs(int(d)),int(n),int(d))

            if best is None or key<best[0]:
                best=(key,n,int(d),int(sup[n]),int(E[n]))

        _,n,d,s,e=best
        return n,d,s,e

    def estimate_batch(self,Q):
        n=len(Q)
        selected=np.empty(n,dtype=np.int64)
        d=np.empty(n,dtype=np.int64)
        support=np.empty(n,dtype=np.int64)
        energy=np.empty(n,dtype=np.int64)

        for i,q in enumerate(Q):
            a,b,c,e=self.estimate_one(q)
            selected[i]=a; d[i]=b; support[i]=c; energy[i]=e

        return selected,d,support,energy


def make_trajectories(X,y):
    frames=[]
    labels=[]
    traj_ids=[]
    true_d=[]
    source_ids=[]
    times=[]

    for src,(x,yy) in enumerate(zip(X,y)):
        for tid,(name,ds) in enumerate(TRAJECTORIES):
            for t,d in enumerate(ds):
                frames.append(np.roll(x,int(d)))
                labels.append(int(yy))
                traj_ids.append(tid)
                true_d.append(int(d))
                source_ids.append(src)
                times.append(t)

    return {
        "frames":np.asarray(frames,dtype=np.uint8),
        "labels":np.asarray(labels,dtype=np.int64),
        "traj_ids":np.asarray(traj_ids,dtype=np.int64),
        "true_d":np.asarray(true_d,dtype=np.int64),
        "source_ids":np.asarray(source_ids,dtype=np.int64),
        "times":np.asarray(times,dtype=np.int64),
    }


def sequence_derivatives(d):
    d=np.asarray(d,dtype=np.int64)
    v=np.diff(d)
    a=np.diff(v)
    return v,a


def exact_temporal_gate():
    # Derivative algebra on every declared trajectory.
    for name,ds in TRAJECTORIES:
        d=np.asarray(ds,dtype=np.int64)
        v,a=sequence_derivatives(d)
        assert len(v)==T-1
        assert len(a)==T-2

        if name.startswith("CV_"):
            assert np.all(v==v[0])
            assert np.all(a==0)

    # Same final position can have different histories.
    # CV_+1 -> final +4; ACC_2_-1 -> final +2 (not same).
    # Find all equal-final pairs automatically and require history distinct.
    equal_pairs=0
    for i,(ni,di) in enumerate(TRAJECTORIES):
        for j,(nj,dj) in enumerate(TRAJECTORIES):
            if j<=i: continue
            if di[-1]==dj[-1] and di!=dj:
                vi,ai=sequence_derivatives(di)
                vj,aj=sequence_derivatives(dj)
                assert not (np.array_equal(vi,vj) and np.array_equal(ai,aj))
                equal_pairs+=1

    return {"pass":True,"trajectory_count":len(TRAJECTORIES),"equal_final_distinct_history_pairs":equal_pairs}


gate=exact_temporal_gate()

# Fixed clean split.
ds=load_digits()
X=(np.asarray(ds.data,dtype=np.int64)*15).astype(np.uint8)
y=np.asarray(ds.target,dtype=np.int64)

outer=StratifiedShuffleSplit(n_splits=1,test_size=.20,random_state=SPLIT_SEED)
tv,te=next(outer.split(X,y))
inner=StratifiedShuffleSplit(n_splits=1,test_size=.25,random_state=SPLIT_SEED+1)
trr,var=next(inner.split(X[tv],y[tv]))
tr=tv[trr]; va=tv[var]

Xtr=X[tr];ytr=y[tr]
Xva=X[va];yva=y[va]
Xte=X[te];yte=y[te]

traj=make_trajectories(Xte,yte)
Q=traj["frames"]

estimator=MovementEstimator(Xtr)
selected,est_d,support,energy=estimator.estimate_batch(Q)

# Estimator convention: aligned train roll(-d_est) ~= query=roll(clean,true_d),
# hence d_est should equal -true_d.
true_inverse=-traj["true_d"]

# Canonicalize moved observations back to clean coordinates.
C=np.empty_like(Q)
for i,(q,de) in enumerate(zip(Q,est_d)):
    C[i]=np.roll(q,int(de))

# Reshape temporal state by source, trajectory, time.
S=len(Xte); J=len(TRAJECTORIES)
est_grid=est_d.reshape(S,J,T)
true_grid=true_inverse.reshape(S,J,T)

# Derivative recovery counts.
disp_correct=int((est_grid==true_grid).sum())
disp_total=est_grid.size

vel_correct=0; vel_total=0
acc_correct=0; acc_total=0
path_exact=0; path_total=S*J

trajectory_rows=[]

for j,(name,ds_) in enumerate(TRAJECTORIES):
    td=np.asarray([-int(x) for x in ds_],dtype=np.int64)
    tv=np.diff(td)
    ta=np.diff(tv)

    d_ok=v_ok=a_ok=path_ok=0

    for s in range(S):
        ed=est_grid[s,j]
        ev=np.diff(ed)
        ea=np.diff(ev)

        d_ok+=int((ed==td).sum())
        v_ok+=int((ev==tv).sum())
        a_ok+=int((ea==ta).sum())
        path_ok+=int(np.array_equal(ed,td))

    vel_correct+=v_ok; vel_total+=S*(T-1)
    acc_correct+=a_ok; acc_total+=S*(T-2)
    path_exact+=path_ok

    trajectory_rows.append({
        "name":name,
        "true_displacement":list(ds_),
        "true_inverse_observer_state":td.tolist(),
        "true_velocity":tv.tolist(),
        "true_acceleration":ta.tolist(),
        "displacement_correct":d_ok,
        "displacement_total":S*T,
        "velocity_correct":v_ok,
        "velocity_total":S*(T-1),
        "acceleration_correct":a_ok,
        "acceleration_total":S*(T-2),
        "exact_path_count":path_ok,
        "path_total":S,
    })


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

    raw_pred=net.predict(Q)
    canon_pred=net.predict(C)

    raw_correct=int((raw_pred==traj["labels"]).sum())
    canon_correct=int((canon_pred==traj["labels"]).sum())
    clean_correct=int((net.predict(Xte)==yte).sum())

    per_seed.append({
        "seed":seed,
        "local_selection":sel,
        "clean_test_correct":clean_correct,
        "clean_test_total":len(yte),
        "trajectory_frame_raw_correct":raw_correct,
        "trajectory_frame_canonicalized_correct":canon_correct,
        "trajectory_frame_total":len(Q),
        "delta_correct":canon_correct-raw_correct,
    })

    print(
        "SEED",seed,
        "CLEAN",clean_correct,"/",len(yte),
        "RAW",raw_correct,"/",len(Q),
        "CANON",canon_correct,"/",len(Q),
        "DELTA",canon_correct-raw_correct,
        flush=True,
    )


raw_sum=sum(r["trajectory_frame_raw_correct"] for r in per_seed)
canon_sum=sum(r["trajectory_frame_canonicalized_correct"] for r in per_seed)

report={
    "model":"v15-U-observer-temporal-movement",
    "gate":gate,
    "observer_state":{
        "displacement":"estimated inverse logical GEN7 displacement d_t",
        "velocity":"d_t-d_(t-1)",
        "acceleration":"v_t-v_(t-1)",
        "logical_to_physical":"one logical step corresponds to one GEN7 physical-row step",
    },
    "trajectory_definitions":[{"name":n,"d":list(d)} for n,d in TRAJECTORIES],
    "estimation":{
        "search_radius":SEARCH_RADIUS,
        "displacement_correct":disp_correct,
        "displacement_total":disp_total,
        "velocity_correct":vel_correct,
        "velocity_total":vel_total,
        "acceleration_correct":acc_correct,
        "acceleration_total":acc_total,
        "exact_path_count":path_exact,
        "path_total":path_total,
        "support_mean":float(support.mean()),
        "energy_mean":float(energy.mean()),
    },
    "trajectory_breakdown":trajectory_rows,
    "classification":{
        "per_seed":per_seed,
        "aggregate_raw_correct":raw_sum,
        "aggregate_canonicalized_correct":canon_sum,
        "aggregate_total":len(MODEL_SEEDS)*len(Q),
        "delta_correct":canon_sum-raw_sum,
    },
    "claim_boundary":(
        "v15 gives U-Observer a controlled temporal role: displacement, velocity, and "
        "acceleration over GEN7 logical time. The trajectories are synthetic and the "
        "observer is not yet the final physical MPRC U law or INFORMATION codec."
    ),
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"u_observer_temporal_movement_v15.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
