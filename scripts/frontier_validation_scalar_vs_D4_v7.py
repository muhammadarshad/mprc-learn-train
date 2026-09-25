"""v7 — Frontier validation: scalar residual vs D4 movement law.

This returns to the strongest architecture before the destructive v4-v6 ablations.

ONLY the parameter-movement memory changes.

Shared between S and D4:
- same 64-byte input
- same trainable Z256 LUT
- same integer linear evidence W,b
- same integer multiclass margin credit
- same denominator 1000
- same initialization
- same epoch sample order
- no softmax
- no floating learning-rate state

Difference:
S:
    one signed residual per parameter; opposite unresolved motion cancels.
D4:
    four residual memories (+U,+D,-U,-D); direction and parameter polarity survive.

Scientific correction relative to v2/v3:
- fixed stratified TRAIN / VALIDATION / untouched TEST split
- checkpoint epoch chosen from VALIDATION only
- TEST evaluated exactly once per trained branch after checkpoint restore
- seven paired seeds; no seed is selected as "winner"
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit

DEN=1000
SPLIT_SEED=20260925
MODEL_SEEDS=(7,19,31,43,59,73,97)
EPOCHS=60
BATCH=64
MARGIN=16
NCLASS=10
FDIM=64


def trunc_div_array(a,q):
    a=np.asarray(a,dtype=np.int64)
    out=np.empty_like(a)
    pos=a>=0
    out[pos]=a[pos]//q
    out[~pos]=-((-a[~pos])//q)
    return out


def signed_coord(value,ring):
    v=np.asarray(value,dtype=np.int64)
    if ring:
        return 128-(v&255)
    return v


class ScalarResidualParam:
    def __init__(self,value,ring=False):
        self.value=np.asarray(value,dtype=np.int64).copy()
        self.residual=np.zeros_like(self.value)
        self.ring=bool(ring)
        self.emitted_abs=0

    def step(self,grad):
        self.residual+=np.asarray(grad,dtype=np.int64)
        k=trunc_div_array(self.residual,DEN)
        self.residual-=k*DEN
        self.value-=k
        if self.ring:
            self.value&=255
        self.emitted_abs+=int(np.abs(k).sum())
        assert np.all(np.abs(self.residual)<DEN)


class Directional4Param:
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


def movement_gate():
    # Same net unresolved displacement, different path memory.
    s=ScalarResidualParam(np.asarray([0]))
    s.step(np.asarray([-999]))
    s.step(np.asarray([+999]))
    assert int(s.value[0])==0
    assert int(s.residual[0])==0

    d=Directional4Param(np.asarray([1]))
    d.step(np.asarray([-999]))
    d.step(np.asarray([+999]))
    assert int(d.value[0])==1
    assert d.residual[0].tolist()==[999,999,0,0]

    # Exact emission threshold.
    d.step(np.asarray([-1]))
    assert int(d.value[0])==2
    d.step(np.asarray([+1]))
    assert int(d.value[0])==1

    return {
        "pass":True,
        "denominator":DEN,
        "bins":["+U","+D","-U","-D"],
        "opposite_999_scalar_residual":int(s.residual[0]),
        "opposite_999_D4_residual":d.residual[0].tolist(),
    }


class Net:
    def __init__(self,mode,W0,b0,L0):
        P=ScalarResidualParam if mode=="S" else Directional4Param
        self.mode=mode
        self.L=P(L0,ring=True)
        self.W=P(W0,ring=False)
        self.b=P(b0,ring=False)

    def forward(self,idx):
        ring=self.L.value[idx]
        a=128-ring
        s=a@self.W.value+self.b.value
        return a,s

    def evaluate(self,idx,y):
        _,s=self.forward(idx)
        p=s.argmax(axis=1)
        return int((p==y).sum()),len(y)

    def values_checkpoint(self):
        return {
            "L":self.L.value.copy(),
            "W":self.W.value.copy(),
            "b":self.b.value.copy(),
        }

    def restore_values(self,ckpt):
        self.L.value=ckpt["L"].copy()
        self.W.value=ckpt["W"].copy()
        self.b.value=ckpt["b"].copy()

    def train_select_on_validation(self,Xtr,ytr,Xval,yval,epoch_orders):
        best_val=(-1,0)
        best_ckpt=None
        history=[]

        for epn,order in enumerate(epoch_orders,1):
            violations=0

            for st in range(0,len(order),BATCH):
                ids=order[st:st+BATCH]
                idx=Xtr[ids]
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

                # IDENTICAL credit calculation for S and D4.
                gW=a.T@ds
                gb=ds.sum(axis=0,dtype=np.int64)
                ga=ds@self.W.value.T
                gL=np.zeros(256,dtype=np.int64)
                np.add.at(gL,idx.reshape(-1),(-ga).reshape(-1))

                self.W.step(gW)
                self.b.step(gb)
                self.L.step(gL)

                self.W.value=np.clip(self.W.value,-127,127)
                self.b.value=np.clip(self.b.value,-32767,32767)

            tr=self.evaluate(Xtr,ytr)
            va=self.evaluate(Xval,yval)

            # VALIDATION ONLY selects checkpoint. Earliest epoch wins ties.
            if va[0]>best_val[0]:
                best_val=(va[0],epn)
                best_ckpt=self.values_checkpoint()

            history.append({
                "epoch":epn,
                "violations":int(violations),
                "train_correct":tr[0],
                "train_total":tr[1],
                "validation_correct":va[0],
                "validation_total":va[1],
            })

        assert best_ckpt is not None
        self.restore_values(best_ckpt)
        return {
            "selected_epoch":best_val[1],
            "selected_validation":[best_val[0],len(yval)],
            "history":history,
        }


# -----------------------
# Fixed untouched split.
# -----------------------
ds=load_digits()
X=(np.asarray(ds.data,dtype=np.int64)*15).astype(np.uint8)
y=np.asarray(ds.target,dtype=np.int64)

outer=StratifiedShuffleSplit(n_splits=1,test_size=0.20,random_state=SPLIT_SEED)
trainval_ix,test_ix=next(outer.split(X,y))
Xtv=X[trainval_ix];ytv=y[trainval_ix]
Xtest=X[test_ix];ytest=y[test_ix]

inner=StratifiedShuffleSplit(n_splits=1,test_size=0.25,random_state=SPLIT_SEED+1)
tr_rel,val_rel=next(inner.split(Xtv,ytv))
train_ix=trainval_ix[tr_rel]
val_ix=trainval_ix[val_rel]

Xtr=X[train_ix];ytr=y[train_ix]
Xval=X[val_ix];yval=y[val_ix]

assert len(set(train_ix)&set(val_ix))==0
assert len(set(train_ix)&set(test_ix))==0
assert len(set(val_ix)&set(test_ix))==0
assert len(train_ix)+len(val_ix)+len(test_ix)==len(X)

gate=movement_gate()

per_seed=[]
paired_deltas=[]

for seed in MODEL_SEEDS:
    init_rng=np.random.default_rng(seed)
    W0=init_rng.integers(-2,3,size=(FDIM,NCLASS),dtype=np.int64)
    b0=np.zeros(NCLASS,dtype=np.int64)
    L0=np.arange(256,dtype=np.int64)

    order_rng=np.random.default_rng(seed+100000)
    epoch_orders=[order_rng.permutation(len(Xtr)) for _ in range(EPOCHS)]

    seed_result={"seed":seed}

    for mode in ("S","D4"):
        net=Net(mode,W0,b0,L0)
        selected=net.train_select_on_validation(Xtr,ytr,Xval,yval,epoch_orders)

        # FIRST AND ONLY TEST EVALUATION for this branch.
        test=net.evaluate(Xtest,ytest)

        telemetry={}
        if mode=="S":
            telemetry={
                "L_emitted_abs":net.L.emitted_abs,
                "W_emitted_abs":net.W.emitted_abs,
                "b_emitted_abs":net.b.emitted_abs,
            }
        else:
            telemetry={
                "L_emitted":net.L.emitted.tolist(),
                "W_emitted":net.W.emitted.tolist(),
                "b_emitted":net.b.emitted.tolist(),
            }

        seed_result[mode]={
            **selected,
            "test_once":[test[0],test[1]],
            "telemetry":telemetry,
        }

        print(
            "SEED",seed,mode,
            "VAL",selected["selected_validation"],
            "EPOCH",selected["selected_epoch"],
            "TEST_ONCE",test,
            flush=True,
        )

    delta=seed_result["D4"]["test_once"][0]-seed_result["S"]["test_once"][0]
    seed_result["paired_test_delta_correct_D4_minus_S"]=delta
    paired_deltas.append(delta)
    per_seed.append(seed_result)


S_counts=[r["S"]["test_once"][0] for r in per_seed]
D_counts=[r["D4"]["test_once"][0] for r in per_seed]
wins=sum(d>0 for d in paired_deltas)
ties=sum(d==0 for d in paired_deltas)
losses=sum(d<0 for d in paired_deltas)

total_test=len(ytest)
sum_S=sum(S_counts)
sum_D=sum(D_counts)

report={
    "model":"v7-frontier-validation-scalar-vs-D4",
    "gate":gate,
    "architecture":"64-byte input -> trainable Z256 LUT -> integer linear evidence -> margin SELECT",
    "controlled_difference":"parameter movement memory only",
    "shared":{
        "denominator":DEN,
        "margin":MARGIN,
        "epochs":EPOCHS,
        "batch":BATCH,
        "softmax":False,
        "float_learning_rate":False,
        "credit_calculation":"integer margin; gW=a.T@ds, gb=sum(ds), gL=accumulate(-(ds@W.T))",
    },
    "split":{
        "method":"fixed stratified 60/20/20",
        "split_seed":SPLIT_SEED,
        "train":len(Xtr),
        "validation":len(Xval),
        "test_untouched":len(Xtest),
        "test_used_for_epoch_selection":False,
        "test_evaluations_per_branch":1,
    },
    "paired_model_seeds":list(MODEL_SEEDS),
    "per_seed":per_seed,
    "aggregate":{
        "scalar_test_correct_sum":sum_S,
        "D4_test_correct_sum":sum_D,
        "paired_total_predictions":len(MODEL_SEEDS)*total_test,
        "D4_minus_scalar_correct_sum":sum_D-sum_S,
        "D4_seed_wins":wins,
        "ties":ties,
        "D4_seed_losses":losses,
        "paired_deltas_correct":paired_deltas,
        "scalar_accuracy_exact":f"{sum_S}/{len(MODEL_SEEDS)*total_test}",
        "D4_accuracy_exact":f"{sum_D}/{len(MODEL_SEEDS)*total_test}",
        "D4_minus_scalar_accuracy_exact":str(Fraction(sum_D-sum_S,len(MODEL_SEEDS)*total_test)),
    },
    "decision_rule":(
        "Treat D4 as a supported movement-law improvement only if the gain is paired "
        "and reasonably stable across seeds. Do not select a seed or epoch from TEST."
    ),
    "claim_boundary":(
        "This isolates scalar-vs-D4 movement memory while retaining the same conventional "
        "integer credit calculation. It does not claim to eliminate credit propagation; "
        "it tests whether MPRC directional/polarity movement is a better optimizer state."
    ),
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"frontier_validation_scalar_vs_D4_v7.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
