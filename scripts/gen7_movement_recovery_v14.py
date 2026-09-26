"""v14 — MPRC GEN7 movement recovery benchmark.

Why this benchmark exists
=========================
v13 selected displacement radius R=0 on every seed. That benchmark was static,
so movement had no job to do.

v14 introduces a controlled movement variable in the SAME 64-state logical
GEN7 orbit used by the attention probes.

One logical roll by d means movement by d generator steps:
    physical row displacement = 7*d mod 64

Train:
    clean digits only.

Validation/Test movement:
    each clean observation is rolled in logical orbit order by
        d in {-7,-4,-2,-1,+1,+2,+4,+7}

No moved sample is used to train the local classifier.

Recovery
========
Given moved query q and clean TRAIN memory x_n, search displacement d' in a
predeclared radius and score:

    support(n,d') = count_f [ q[f] == x_n[(f-d') mod64] ]

Tie-break:
    higher support
    lower circular MEASURE
    smaller |d'|
    sample id

If aligned TRAIN state is:
    roll(x_n, -d') ~= q

then the query is canonicalized by:
    q_canon = roll(q, d')

The frozen local classifier then classifies q_canon.

This tests movement estimation, not augmentation:
- local clean model is unchanged
- retrieval is label-free
- recovery radius selected on moved VALIDATION only
- TEST evaluated once after validation selection

R=0 is the no-movement-recovery control.
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

TRUE_DISPLACEMENTS=(-7,-4,-2,-1,1,2,4,7)
RADII=(0,1,2,4,7)


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

            p=self.predict(Xva)
            ac=int((p==yva).sum())
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


def make_moved(X,y):
    Xm=[];ym=[];dm=[];source=[]
    for i,(x,yy) in enumerate(zip(X,y)):
        for d in TRUE_DISPLACEMENTS:
            Xm.append(np.roll(x,int(d)))
            ym.append(int(yy))
            dm.append(int(d))
            source.append(i)
    return (
        np.asarray(Xm,dtype=np.uint8),
        np.asarray(ym,dtype=np.int64),
        np.asarray(dm,dtype=np.int64),
        np.asarray(source,dtype=np.int64),
    )


class MovementEstimator:
    def __init__(self,Xmem):
        self.X=np.asarray(Xmem,dtype=np.uint8)
        self.N=len(self.X)

    def estimate_one(self,q,R):
        best=None
        for d in range(-R,R+1):
            aligned=np.roll(self.X,-int(d),axis=1)
            sup=(aligned==q[None,:]).sum(axis=1,dtype=np.int64)
            E=cdist_rows(q,aligned)

            # best sample under this d
            n=int(np.lexsort((np.arange(self.N),E,-sup))[0])
            key=(-int(sup[n]),int(E[n]),abs(int(d)),int(n),int(d))
            if best is None or key<best[0]:
                best=(key,n,int(d),int(sup[n]),int(E[n]))

        _,n,d,s,e=best
        return n,d,s,e

    def canonicalize_batch(self,Q,R):
        C=np.empty_like(Q)
        est=np.empty(len(Q),dtype=np.int64)
        selected=np.empty(len(Q),dtype=np.int64)
        support=np.empty(len(Q),dtype=np.int64)
        energy=np.empty(len(Q),dtype=np.int64)

        for i,q in enumerate(Q):
            n,d,s,e=self.estimate_one(q,R)
            selected[i]=n
            est[i]=d
            support[i]=s
            energy[i]=e

            # aligned train sample roll(x,-d) approximates q,
            # hence canonical query is roll(q,d).
            C[i]=np.roll(q,int(d))

        return C,selected,est,support,energy


def exact_recovery_gate():
    # If query is an exact roll of a stored state, search radius containing that
    # displacement must be able to find an exact zero-energy alignment.
    rng=np.random.default_rng(14)
    X=rng.integers(0,256,size=(12,FDIM),dtype=np.uint8)
    est=MovementEstimator(X)

    for n in range(len(X)):
        for true_d in TRUE_DISPLACEMENTS:
            q=np.roll(X[n],true_d)
            _,d,s,e=est.estimate_one(q,7)
            assert e==0
            assert s==FDIM
            # Definition has aligned x roll(-d)=q, therefore d=-true_d mod the
            # small signed search range for these non-wrapping values.
            assert d==-true_d

    return {
        "pass":True,
        "stored_states":len(X),
        "displacements":list(TRUE_DISPLACEMENTS),
    }


gate=exact_recovery_gate()

# Fixed clean 60/20/20 split.
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

Mva,YMva,Dva,Sva=make_moved(Xva,yva)
Mte,YMte,Dte,Ste=make_moved(Xte,yte)

movement=MovementEstimator(Xtr)

# Model-independent movement recovery precomputed for each R.
recovery={}
for R in RADII:
    Cva,Selva,Estva,Supva,Eva=movement.canonicalize_batch(Mva,R)
    Cte,Selte,Estte,Supte,Ete=movement.canonicalize_batch(Mte,R)
    recovery[R]={
        "val":(Cva,Selva,Estva,Supva,Eva),
        "test":(Cte,Selte,Estte,Supte,Ete),
    }

per_seed=[]
deltas=[]

for seed in MODEL_SEEDS:
    rng=np.random.default_rng(seed)
    W0=rng.integers(-2,3,size=(FDIM,NCLASS),dtype=np.int64)
    b0=np.zeros(NCLASS,dtype=np.int64)
    L0=np.arange(256,dtype=np.int64)

    rr=np.random.default_rng(seed+100000)
    orders=[rr.permutation(len(Xtr)) for _ in range(EPOCHS)]

    net=LocalNet(W0,b0,L0)
    sel=net.train_select_validation(Xtr,ytr,Xva,yva,orders)

    # Select movement radius on MOVED VALIDATION only.
    best=None
    val_rows={}

    for R in RADII:
        Cva,_,Estva,Supva,Eva=recovery[R]["val"]
        p=net.predict(Cva)
        ac=int((p==YMva).sum())

        # known synthetic displacement diagnostic; not used in selection key
        exact_d=int((Estva==-Dva).sum())

        val_rows[str(R)]={
            "correct":ac,
            "total":len(YMva),
            "exact_displacement_recovery":exact_d,
            "support_mean":float(Supva.mean()),
            "energy_mean":float(Eva.mean()),
        }

        key=(ac,-R)
        if best is None or key>best[0]:
            best=(key,R)

    Rsel=best[1]

    # TEST performance first accessed after R is fixed.
    local_moved_pred=net.predict(Mte)
    local_moved_correct=int((local_moved_pred==YMte).sum())

    Cte,Selte,Estte,Supte,Ete=recovery[Rsel]["test"]
    recovered_pred=net.predict(Cte)
    recovered_correct=int((recovered_pred==YMte).sum())

    # Clean-test sanity is also fixed, not selected.
    clean_correct=int((net.predict(Xte)==yte).sum())

    exact_d=int((Estte==-Dte).sum())
    delta=recovered_correct-local_moved_correct
    deltas.append(delta)

    by_true_d={}
    for d in TRUE_DISPLACEMENTS:
        m=Dte==d
        by_true_d[str(d)]={
            "count":int(m.sum()),
            "local_moved_correct":int((local_moved_pred[m]==YMte[m]).sum()),
            "recovered_correct":int((recovered_pred[m]==YMte[m]).sum()),
            "exact_estimated_inverse_displacement":int((Estte[m]==-d).sum()),
            "estimated_d_mean":float(Estte[m].mean()),
        }

    per_seed.append({
        "seed":seed,
        "local_selection":sel,
        "selected_radius":Rsel,
        "validation_by_radius":val_rows,
        "test":{
            "clean_correct":clean_correct,
            "clean_total":len(yte),
            "moved_local_correct":local_moved_correct,
            "moved_recovered_correct":recovered_correct,
            "moved_total":len(YMte),
            "delta_correct":delta,
            "exact_inverse_displacement_recovery":exact_d,
            "support_mean":float(Supte.mean()),
            "energy_mean":float(Ete.mean()),
            "by_true_displacement":by_true_d,
        },
    })

    print(
        "SEED",seed,
        "R",Rsel,
        "CLEAN",clean_correct,"/",len(yte),
        "MOVED_LOCAL",local_moved_correct,"/",len(YMte),
        "RECOVERED",recovered_correct,"/",len(YMte),
        "DELTA",delta,
        "D_EXACT",exact_d,"/",len(YMte),
        flush=True,
    )


moved_local_sum=sum(r["test"]["moved_local_correct"] for r in per_seed)
recovered_sum=sum(r["test"]["moved_recovered_correct"] for r in per_seed)
clean_sum=sum(r["test"]["clean_correct"] for r in per_seed)

report={
    "model":"v14-GEN7-movement-recovery",
    "gate":gate,
    "training":"clean TRAIN only; no moved augmentation",
    "movement":{
        "logical_displacements":list(TRUE_DISPLACEMENTS),
        "physical_interpretation":"one logical step = one GEN7 physical-row step on the active orbit",
        "canonicalization":"if roll(train,-d*) ~= moved query, use roll(query,d*)",
    },
    "radius_candidates":list(RADII),
    "selection":"radius selected on moved VALIDATION only",
    "test_tuning":False,
    "per_seed":per_seed,
    "aggregate":{
        "clean_correct_sum":clean_sum,
        "clean_total":len(MODEL_SEEDS)*len(yte),
        "moved_local_correct_sum":moved_local_sum,
        "moved_recovered_correct_sum":recovered_sum,
        "moved_total":len(MODEL_SEEDS)*len(YMte),
        "recovery_delta_correct":recovered_sum-moved_local_sum,
        "seed_wins":sum(d>0 for d in deltas),
        "ties":sum(d==0 for d in deltas),
        "seed_losses":sum(d<0 for d in deltas),
    },
    "claim_boundary":(
        "v14 is a controlled GEN7-orbit movement stress test. The imposed logical "
        "roll is synthetic movement, not yet physical U-observer time dynamics. "
        "It tests whether displacement can be estimated and used as state when motion exists."
    ),
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"gen7_movement_recovery_v14.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
