"""v2 — Scalar residual learning vs directional/polarity residual learning.

Question
========
A rational step eta=1/1000 can be represented exactly without floats. But the
usual scalar residual collapses opposite sub-quantum motions before they emit.

This probe tests the user's proposed higher-dimensional memory:

    (+U, +D, -U, -D)

where:
- U / D = requested parameter movement direction,
- + / - = parameter polarity at the time the movement was observed.

Thus opposite motions can have zero NET displacement while their directional
history remains present in separate residual bins.

This is a candidate learning law, not yet a frozen MPRC theorem.

Controlled A/B
==============
Same split, same seeds, same integer multiclass-margin objective, same trainable
Z256 LUT, no softmax, no float LR.

Forward graphs:
  Q  : local/query only
  W2 : exact pair (query, selected global context)

Optimizers:
  S  : scalar residual accumulator
  D4 : four-bin directional/polarity residual accumulator

Branches:
  Q-S, Q-D4, W2-S, W2-D4
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits

from mprc_structural.global_attention import HV, Relation16TransposePolicy
from mprc_structural.manifold import W as MAN_W
from mprc_structural.attention import generator_orbit64

DEN=1000
SEED=20260925
EPOCHS=40
BATCH=64
MARGIN=16
NCLASS=10
FDIM=64
COL=1

ORBIT=generator_orbit64()
SITE=np.asarray([int(ORBIT[t])*MAN_W+COL for t in range(FDIM)],dtype=np.int64)


def trunc_div_array(a,q):
    a=np.asarray(a,dtype=np.int64)
    out=np.empty_like(a)
    pos=a>=0
    out[pos]=a[pos]//q
    out[~pos]=-((-a[~pos])//q)
    return out


def signed_coord(value:np.ndarray,ring:bool)->np.ndarray:
    v=np.asarray(value,dtype=np.int64)
    if ring:
        # MPRC origin 128: below origin positive, above origin negative.
        return 128-(v&255)
    return v


class ScalarResidualParam:
    def __init__(self,value,ring=False):
        self.value=np.asarray(value,dtype=np.int64).copy()
        self.residual=np.zeros_like(self.value)
        self.ring=bool(ring)
        self.emitted_abs=0

    def step(self,grad):
        self.residual += np.asarray(grad,dtype=np.int64)
        k=trunc_div_array(self.residual,DEN)
        self.residual -= k*DEN
        self.value -= k
        if self.ring:
            self.value &= 255
        self.emitted_abs += int(np.abs(k).sum())


class Directional4Param:
    """Four independent residual memories: +U,+D,-U,-D.

    Parameter request is delta=-gradient.
    U := positive parameter movement.
    D := negative parameter movement.
    Polarity is observed BEFORE applying this step.

    Residual bins do not cancel one another before emission.
    """

    def __init__(self,value,ring=False):
        self.value=np.asarray(value,dtype=np.int64).copy()
        self.residual=np.zeros(self.value.shape+(4,),dtype=np.int64)
        self.ring=bool(ring)
        self.emitted=np.zeros(4,dtype=np.int64)

    def step(self,grad):
        g=np.asarray(grad,dtype=np.int64)
        req=-g
        mag=np.abs(req)
        coord=signed_coord(self.value,self.ring)

        up=req>0
        down=req<0
        plus=coord>=0
        minus=~plus

        masks=(plus&up, plus&down, minus&up, minus&down)
        delta=np.zeros_like(self.value)

        for d,mask in enumerate(masks):
            if not np.any(mask):
                continue
            r=self.residual[...,d]
            r[mask] += mag[mask]
            k=np.zeros_like(r)
            k[mask]=r[mask]//DEN
            r[mask]-=k[mask]*DEN
            self.emitted[d]+=int(k.sum())
            if d in (0,2):   # U
                delta += k
            else:            # D
                delta -= k

        self.value += delta
        if self.ring:
            self.value &= 255

        assert np.all((self.residual>=0)&(self.residual<DEN))


def directional_survival_gate():
    # Scalar cancellation loses the path.
    s=ScalarResidualParam(np.asarray([0]))
    s.step(np.asarray([-999]))  # requested +999
    s.step(np.asarray([+999]))  # requested -999
    assert int(s.value[0])==0
    assert int(s.residual[0])==0

    # Directional memory retains both unresolved paths.
    d=Directional4Param(np.asarray([1]))  # positive polarity
    d.step(np.asarray([-999]))            # +U
    d.step(np.asarray([+999]))            # +D
    assert int(d.value[0])==1
    assert int(d.residual[0,0])==999
    assert int(d.residual[0,1])==999

    # One more count in each direction emits independently.
    d.step(np.asarray([-1]))
    assert int(d.value[0])==2
    d.step(np.asarray([+1]))
    assert int(d.value[0])==1

    # Ring polarity is relative to 128.
    r=Directional4Param(np.asarray([64]),ring=True)   # positive side
    r.step(np.asarray([-1000]))                       # +U => bin0
    assert int(r.emitted[0])==1

    r2=Directional4Param(np.asarray([192]),ring=True) # negative side
    r2.step(np.asarray([-1000]))                      # -U => bin2
    assert int(r2.emitted[2])==1

    return {
        "scalar_opposite_999_residual_after_cancel":int(s.residual[0]),
        "directional_plusU_residual":999,
        "directional_plusD_residual":999,
        "ring_positive_U_bin":0,
        "ring_negative_U_bin":2,
        "pass":True,
    }


def cdist_rows(q,X):
    qa=q.astype(np.int16)[None,:]
    xa=X.astype(np.int16)
    ab=(qa-xa)&255
    ba=(xa-qa)&255
    return np.minimum(ab,ba).sum(axis=1,dtype=np.int64)


class ActiveGlobalTranspose:
    def __init__(self,Xmem):
        self.X=np.asarray(Xmem,dtype=np.uint8)
        self.N=len(self.X)
        buckets=[[] for _ in range(256)]
        for n,row in enumerate(self.X):
            base=n*HV
            for f,p0 in enumerate(row):
                buckets[int(p0)].append((base+int(SITE[f]),n,f))
        self.buckets=buckets

    def candidate_samples(self,q,exclude=None):
        seen=set()
        pol=Relation16TransposePolicy()
        for p0 in q:
            for p in pol.states(int(p0)):
                for g,n,f in self.buckets[p]:
                    if exclude is not None and n==exclude:
                        continue
                    seen.add(n)
        if not seen:
            seen=set(range(self.N))
            if exclude is not None: seen.discard(exclude)
        return np.asarray(sorted(seen),dtype=np.int64)


def contexts(Xq,index,training):
    out=np.empty_like(Xq)
    sel=np.empty(len(Xq),dtype=np.int64)
    for i,q in enumerate(Xq):
        cand=index.candidate_samples(q,exclude=(i if training else None))
        E=cdist_rows(q,index.X[cand])
        order=np.lexsort((cand,E))
        b=int(cand[order[0]])
        out[i]=index.X[b]
        sel[i]=b
    return out,sel


def indices(kind,q,c):
    if kind=="Q":
        return q
    if kind=="W2":
        return np.concatenate([q,c],axis=1)
    raise KeyError(kind)


class Net:
    def __init__(self,fdim,mode,seed):
        P=ScalarResidualParam if mode=="S" else Directional4Param
        self.mode=mode
        self.L=P(np.arange(256,dtype=np.int64),ring=True)
        rng=np.random.default_rng(seed)
        self.W=P(rng.integers(-2,3,size=(fdim,NCLASS),dtype=np.int64),ring=False)
        self.b=P(np.zeros(NCLASS,dtype=np.int64),ring=False)
        self.L0=self.L.value.copy()

    def forward(self,idx):
        ring=self.L.value[idx]
        a=128-ring
        return a,a@self.W.value+self.b.value

    def evaluate(self,idx,y):
        _,s=self.forward(idx)
        p=s.argmax(axis=1)
        return int((p==y).sum()),len(y)

    def train(self,itr,ytr,ite,yte):
        rng=np.random.default_rng(SEED+900+itr.shape[1]+(0 if self.mode=="S" else 1))
        hist=[]
        best=(0,0)
        for epn in range(EPOCHS):
            order=rng.permutation(len(itr))
            vio=0
            for st in range(0,len(order),BATCH):
                ids=order[st:st+BATCH]
                idx=itr[ids]; yy=ytr[ids]
                a,s=self.forward(idx)
                ds=np.zeros_like(s,dtype=np.int64)
                for i in range(len(ids)):
                    yi=int(yy[i]); sy=int(s[i,yi]); v=0
                    for c in range(NCLASS):
                        if c==yi: continue
                        if int(s[i,c])+MARGIN>sy:
                            ds[i,c]+=1;v+=1
                    ds[i,yi]-=v;vio+=v

                gW=a.T@ds
                gb=ds.sum(axis=0,dtype=np.int64)
                ga=ds@self.W.value.T
                gL=np.zeros(256,dtype=np.int64)
                np.add.at(gL,idx.reshape(-1),(-ga).reshape(-1))

                self.W.step(gW);self.b.step(gb);self.L.step(gL)
                self.W.value=np.clip(self.W.value,-127,127)
                self.b.value=np.clip(self.b.value,-32767,32767)

            tr=self.evaluate(itr,ytr);te=self.evaluate(ite,yte)
            if te[0]>best[0]:best=(te[0],epn+1)
            hist.append({
                "epoch":epn+1,
                "train_correct":tr[0],"train_total":tr[1],
                "test_correct":te[0],"test_total":te[1],
                "violations":int(vio),
            })
        return {
            "final_test":list(self.evaluate(ite,yte)),
            "best_test":[best[0],len(yte)],
            "best_epoch":best[1],
            "lut_changed":int(np.count_nonzero(self.L.value!=self.L0)),
            "history":hist,
        }


gate=directional_survival_gate()

ds=load_digits()
X=(np.asarray(ds.data,dtype=np.int64)*15).astype(np.uint8)
y=np.asarray(ds.target,dtype=np.int64)

rng=np.random.default_rng(SEED)
order=rng.permutation(len(X))
cut=int(len(order)*.7)
trix=order[:cut];teix=order[cut:]
Xtr=X[trix];ytr=y[trix];Xte=X[teix];yte=y[teix]

index=ActiveGlobalTranspose(Xtr)
Ctr,seltr=contexts(Xtr,index,True)
Cte,selte=contexts(Xte,index,False)
neighbor=int((ytr[selte]==yte).sum())

results={}
for kind in ("Q","W2"):
    Itr=indices(kind,Xtr,Ctr)
    Ite=indices(kind,Xte,Cte)
    for mode in ("S","D4"):
        key=f"{kind}-{mode}"
        net=Net(Itr.shape[1],mode,SEED+(0 if kind=="Q" else 100)+(0 if mode=="S" else 1))
        res=net.train(Itr,ytr,Ite,yte)
        if mode=="D4":
            res["directional_emitted"]={
                "LUT":net.L.emitted.tolist(),
                "W":net.W.emitted.tolist(),
                "b":net.b.emitted.tolist(),
            }
            res["directional_residual_totals"]={
                "LUT":net.L.residual.sum(axis=tuple(range(net.L.residual.ndim-1))).tolist(),
                "W":net.W.residual.sum(axis=tuple(range(net.W.residual.ndim-1))).tolist(),
                "b":net.b.residual.sum(axis=tuple(range(net.b.residual.ndim-1))).tolist(),
            }
        results[key]=res
        print(key,res["final_test"],"best",res["best_test"],"epoch",res["best_epoch"],flush=True)

report={
    "model":"v2-scalar-vs-directional4-learning",
    "survival_gate":gate,
    "directional_definition":{
        "bins":["+U","+D","-U","-D"],
        "U":"positive requested parameter movement",
        "D":"negative requested parameter movement",
        "polarity_ring":"128-value >=0 is + side; <0 is - side",
        "polarity_signed":"parameter >=0 is + side; <0 is - side",
        "important":"opposite sub-quantum paths do not cancel before emission",
    },
    "ga_context":{
        "test_neighbor_label_agreement":[neighbor,len(yte)],
        "labels_used_for_selection":False,
        "W2":"exact pair (query,selected context)",
    },
    "learning":{
        "denominator":DEN,
        "softmax":False,
        "float_learning_rate":False,
        "integer_margin":True,
        "trainable_Z256_LUT":True,
    },
    "results":results,
    "claim_boundary":(
        "D4 is an experimental directional/polarity memory law. This experiment can "
        "show whether retaining path information changes learning behavior relative to "
        "the scalar residual baseline; it does not yet prove D4 is the final MPRC replacement "
        "for gradient descent."
    )
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"scalar_vs_directional4_learning_v2.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
