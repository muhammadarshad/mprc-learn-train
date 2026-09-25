"""v8 — Parallel Global Attention Residual Evidence.

Purpose
=======
v7 showed scalar residual and D4 movement have identical aggregate untouched-test
accuracy across seven paired seeds. Therefore v8 freezes the simpler exact
1/1000 scalar movement law and isolates ATTENTION.

Do NOT collapse query and context.

Stage 1 — LOCAL
---------------
    Q -> trainable Z256 LUT -> integer evidence Wq,bq

Train on TRAIN. Select epoch using VALIDATION only. Freeze.

Stage 2 — ATTENTION RESIDUAL
----------------------------
    Q local states
      -> Relation16TransposePolicy
      -> sparse global occurrences in TRAIN memory
      -> ring MEASURE
      -> selected context C

    C -> SAME FROZEN local LUT -> correction evidence Wg,bg

Final:
    S = S_local(Q) + gate(E_measure) * S_global(C)

Only Wg,bg train in stage 2. The local path cannot be damaged.

The confidence threshold gate uses only ring MEASURE energy. Candidate thresholds
come from TRAIN energies; epoch+threshold are selected on VALIDATION. TEST is
evaluated exactly once after checkpoint selection.

No softmax, no floating learning rate. Attention retrieval is label-free.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit

from mprc_structural.global_attention import HV, Relation16TransposePolicy
from mprc_structural.manifold import W as MAN_W
from mprc_structural.attention import generator_orbit64

DEN=1000
SPLIT_SEED=20260925
MODEL_SEEDS=(7,19,31,43,59)
LOCAL_EPOCHS=60
GLOBAL_EPOCHS=50
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


class ScalarResidualParam:
    def __init__(self,value):
        self.value=np.asarray(value,dtype=np.int64).copy()
        self.residual=np.zeros_like(self.value,dtype=np.int64)
        self.emitted_abs=0

    def step(self,grad):
        self.residual+=np.asarray(grad,dtype=np.int64)
        k=trunc_div_array(self.residual,DEN)
        self.residual-=k*DEN
        self.value-=k
        self.emitted_abs+=int(np.abs(k).sum())
        assert np.all(np.abs(self.residual)<DEN)


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
        policy=Relation16TransposePolicy()
        for p0 in q:
            for p in policy.states(int(p0)):
                for g,n,f in self.buckets[p]:
                    if exclude is not None and n==exclude:
                        continue
                    seen.add(n)
        if not seen:
            seen=set(range(self.N))
            if exclude is not None:
                seen.discard(exclude)
        return np.asarray(sorted(seen),dtype=np.int64)

    def select(self,Xq,training=False):
        C=np.empty_like(Xq)
        selected=np.empty(len(Xq),dtype=np.int64)
        e1=np.empty(len(Xq),dtype=np.int64)
        e2=np.empty(len(Xq),dtype=np.int64)

        for i,q in enumerate(Xq):
            cand=self.candidate_samples(q,exclude=(i if training else None))
            E=cdist_rows(q,self.X[cand])
            order=np.lexsort((cand,E))
            best=int(order[0])
            selected[i]=int(cand[best])
            C[i]=self.X[selected[i]]
            e1[i]=int(E[best])
            e2[i]=int(E[order[1]]) if len(order)>1 else int(E[best])
        return C,selected,e1,e2


class LocalNet:
    def __init__(self,W0,b0,L0):
        self.L=ScalarResidualParam(L0)
        self.W=ScalarResidualParam(W0)
        self.b=ScalarResidualParam(b0)

    def features(self,X):
        return 128-self.L.value[X]

    def scores(self,X):
        a=self.features(X)
        return a,a@self.W.value+self.b.value

    def evaluate(self,X,y):
        _,s=self.scores(X)
        p=s.argmax(axis=1)
        return int((p==y).sum()),len(y)

    def ckpt(self):
        return {
            "L":self.L.value.copy(),
            "W":self.W.value.copy(),
            "b":self.b.value.copy(),
        }

    def restore(self,c):
        self.L.value=c["L"].copy()
        self.W.value=c["W"].copy()
        self.b.value=c["b"].copy()

    def train_select_val(self,Xtr,ytr,Xval,yval,orders):
        best=(-1,0)
        best_ck=None
        hist=[]

        for ep,order in enumerate(orders,1):
            vio=0
            for st in range(0,len(order),BATCH):
                ids=order[st:st+BATCH]
                x=Xtr[ids]; yy=ytr[ids]
                a,s=self.scores(x)
                ds=np.zeros_like(s,dtype=np.int64)

                for i in range(len(ids)):
                    yi=int(yy[i]); sy=int(s[i,yi]); v=0
                    for c in range(NCLASS):
                        if c==yi: continue
                        if int(s[i,c])+MARGIN>sy:
                            ds[i,c]+=1; v+=1
                    ds[i,yi]-=v; vio+=v

                gW=a.T@ds
                gb=ds.sum(axis=0,dtype=np.int64)
                ga=ds@self.W.value.T
                gL=np.zeros(256,dtype=np.int64)
                np.add.at(gL,x.reshape(-1),(-ga).reshape(-1))

                self.W.step(gW); self.b.step(gb); self.L.step(gL)
                self.W.value=np.clip(self.W.value,-127,127)
                self.b.value=np.clip(self.b.value,-32767,32767)
                self.L.value&=255

            tr=self.evaluate(Xtr,ytr)
            va=self.evaluate(Xval,yval)
            if va[0]>best[0]:
                best=(va[0],ep)
                best_ck=self.ckpt()
            hist.append({
                "epoch":ep,
                "violations":vio,
                "train_correct":tr[0],
                "validation_correct":va[0],
                "validation_total":va[1],
            })

        self.restore(best_ck)
        return {
            "selected_epoch":best[1],
            "selected_validation":[best[0],len(yval)],
            "history":hist,
        }


class GlobalCorrection:
    def __init__(self,L_frozen,W0,b0):
        self.L=np.asarray(L_frozen,dtype=np.int64).copy()
        self.W=ScalarResidualParam(W0)
        self.b=ScalarResidualParam(b0)

    def scores(self,C):
        a=128-self.L[C]
        return a,a@self.W.value+self.b.value

    def ckpt(self):
        return {"W":self.W.value.copy(),"b":self.b.value.copy()}

    def restore(self,c):
        self.W.value=c["W"].copy()
        self.b.value=c["b"].copy()


def margin_ds(scores,y):
    ds=np.zeros_like(scores,dtype=np.int64)
    vio=0
    for i in range(len(y)):
        yi=int(y[i]); sy=int(scores[i,yi]); v=0
        for c in range(NCLASS):
            if c==yi: continue
            if int(scores[i,c])+MARGIN>sy:
                ds[i,c]+=1; v+=1
        ds[i,yi]-=v; vio+=v
    return ds,vio


def accuracy_scores(s,y):
    p=s.argmax(axis=1)
    return int((p==y).sum()),len(y)


def make_thresholds(train_e):
    z=np.sort(np.asarray(train_e,dtype=np.int64))
    qs=(25,50,75,90,100)
    out=[]
    for q in qs:
        idx=min(len(z)-1,(len(z)*q)//100)
        out.append(int(z[idx]))
    return sorted(set(out))


def apply_gate(local_s,global_s,e,thr):
    gate=(np.asarray(e)<=int(thr)).astype(np.int64)[:,None]
    return local_s + gate*global_s


def train_global_correction(
    local_net,
    Ctr,ytr,Etr,
    Cval,yval,Eval,
    local_tr_s,local_val_s,
    W0,b0,orders,thresholds
):
    g=GlobalCorrection(local_net.L.value,W0,b0)
    best=(-1,0,None)
    best_ck=None
    hist=[]

    for ep,order in enumerate(orders,1):
        vio=0
        for st in range(0,len(order),BATCH):
            ids=order[st:st+BATCH]
            c=Ctr[ids]; yy=ytr[ids]
            _,gs=g.scores(c)

            # Train correction against FULL combined score. Local branch is frozen.
            combined=local_tr_s[ids]+gs
            ds,v=margin_ds(combined,yy)
            vio+=v

            a=128-g.L[c]
            gW=a.T@ds
            gb=ds.sum(axis=0,dtype=np.int64)
            g.W.step(gW); g.b.step(gb)
            g.W.value=np.clip(g.W.value,-127,127)
            g.b.value=np.clip(g.b.value,-32767,32767)

        _,gval=g.scores(Cval)

        ep_best=(-1,None)
        for thr in thresholds:
            s=apply_gate(local_val_s,gval,Eval,thr)
            ac=accuracy_scores(s,yval)[0]
            if ac>ep_best[0]:
                ep_best=(ac,thr)

        if ep_best[0]>best[0]:
            best=(ep_best[0],ep,ep_best[1])
            best_ck=g.ckpt()

        hist.append({
            "epoch":ep,
            "violations":vio,
            "best_validation_correct":ep_best[0],
            "best_threshold":ep_best[1],
        })

    g.restore(best_ck)
    return g,{
        "selected_validation":[best[0],len(yval)],
        "selected_epoch":best[1],
        "selected_measure_threshold":best[2],
        "history":hist,
    }


# -----------------------
# Fixed stratified split.
# -----------------------
ds=load_digits()
X=(np.asarray(ds.data,dtype=np.int64)*15).astype(np.uint8)
y=np.asarray(ds.target,dtype=np.int64)

outer=StratifiedShuffleSplit(n_splits=1,test_size=0.20,random_state=SPLIT_SEED)
trainval_ix,test_ix=next(outer.split(X,y))
Xtv=X[trainval_ix];ytv=y[trainval_ix]

inner=StratifiedShuffleSplit(n_splits=1,test_size=0.25,random_state=SPLIT_SEED+1)
tr_rel,val_rel=next(inner.split(Xtv,ytv))
train_ix=trainval_ix[tr_rel]
val_ix=trainval_ix[val_rel]

Xtr=X[train_ix];ytr=y[train_ix]
Xval=X[val_ix];yval=y[val_ix]
Xtest=X[test_ix];ytest=y[test_ix]

assert not (set(train_ix)&set(val_ix))
assert not (set(train_ix)&set(test_ix))
assert not (set(val_ix)&set(test_ix))

# -----------------------
# Label-free attention memory from TRAIN only.
# -----------------------
index=ActiveGlobalTranspose(Xtr)
Ctr,seltr,Etr,Etr2=index.select(Xtr,training=True)
Cval,selval,Eval,Eval2=index.select(Xval,training=False)
Ctest,seltest,Etest,Etest2=index.select(Xtest,training=False)

thresholds=make_thresholds(Etr)

per_seed=[]
paired_deltas=[]

for seed in MODEL_SEEDS:
    rng=np.random.default_rng(seed)
    W0=rng.integers(-2,3,size=(FDIM,NCLASS),dtype=np.int64)
    b0=np.zeros(NCLASS,dtype=np.int64)
    L0=np.arange(256,dtype=np.int64)

    order_rng=np.random.default_rng(seed+100000)
    local_orders=[order_rng.permutation(len(Xtr)) for _ in range(LOCAL_EPOCHS)]
    global_orders=[order_rng.permutation(len(Xtr)) for _ in range(GLOBAL_EPOCHS)]

    local=LocalNet(W0,b0,L0)
    local_sel=local.train_select_val(Xtr,ytr,Xval,yval,local_orders)

    # Frozen local scores for correction training/selection.
    _,local_tr_s=local.scores(Xtr)
    _,local_val_s=local.scores(Xval)

    # Correction starts at exact zero evidence.
    GW0=np.zeros((FDIM,NCLASS),dtype=np.int64)
    Gb0=np.zeros(NCLASS,dtype=np.int64)

    glob,glob_sel=train_global_correction(
        local,
        Ctr,ytr,Etr,
        Cval,yval,Eval,
        local_tr_s,local_val_s,
        GW0,Gb0,global_orders,thresholds
    )

    # TEST IS FIRST ACCESSED HERE FOR MODEL PERFORMANCE.
    _,local_test_s=local.scores(Xtest)
    local_test=accuracy_scores(local_test_s,ytest)

    _,global_test_s=glob.scores(Ctest)
    combined_test_s=apply_gate(
        local_test_s,global_test_s,Etest,glob_sel["selected_measure_threshold"]
    )
    combined_test=accuracy_scores(combined_test_s,ytest)

    delta=combined_test[0]-local_test[0]
    paired_deltas.append(delta)

    per_seed.append({
        "seed":seed,
        "local_selection":local_sel,
        "attention_selection":glob_sel,
        "local_test_once":list(local_test),
        "parallel_attention_test_once":list(combined_test),
        "delta_correct":delta,
    })

    print(
        "SEED",seed,
        "LOCAL",local_test,
        "ATTN",combined_test,
        "DELTA",delta,
        "THR",glob_sel["selected_measure_threshold"],
        flush=True,
    )


local_sum=sum(r["local_test_once"][0] for r in per_seed)
attn_sum=sum(r["parallel_attention_test_once"][0] for r in per_seed)
wins=sum(d>0 for d in paired_deltas)
ties=sum(d==0 for d in paired_deltas)
losses=sum(d<0 for d in paired_deltas)

# Test-label diagnostics only AFTER all model selections are complete.
neighbor_label_agreement=int((ytr[seltest]==ytest).sum())

report={
    "model":"v8-parallel-global-attention-residual-evidence",
    "architecture":{
        "local":"Q -> frozen selected local network",
        "attention":"Q -> Relation16 transpose -> TRAIN global occurrences -> ring MEASURE -> selected C",
        "fusion":"S_local(Q) + gated S_global(C)",
        "query_context_collapsed":False,
        "local_branch_modified_by_attention_training":False,
    },
    "learning":{
        "denominator":DEN,
        "movement":"exact scalar residual 1/1000",
        "softmax":False,
        "float_learning_rate":False,
        "local_LUT_trainable_stage1":True,
        "local_path_frozen_stage2":True,
        "global_uses_frozen_local_LUT":True,
    },
    "split":{
        "train":len(Xtr),
        "validation":len(Xval),
        "test_untouched":len(Xtest),
        "epoch_and_threshold_selected_on_validation":True,
        "test_evaluations_per_final_branch":1,
    },
    "attention":{
        "memory":"TRAIN only",
        "labels_used_for_retrieval":False,
        "relation":"Relation16TransposePolicy",
        "measure":"sum of 64 circular Z256 distances",
        "candidate_thresholds_from_train_measure_energy":thresholds,
        "test_neighbor_label_agreement_diagnostic":[neighbor_label_agreement,len(ytest)],
    },
    "paired_model_seeds":list(MODEL_SEEDS),
    "per_seed":per_seed,
    "aggregate":{
        "local_correct_sum":local_sum,
        "parallel_attention_correct_sum":attn_sum,
        "paired_total_predictions":len(MODEL_SEEDS)*len(ytest),
        "attention_minus_local_correct_sum":attn_sum-local_sum,
        "attention_seed_wins":wins,
        "ties":ties,
        "attention_seed_losses":losses,
        "paired_deltas_correct":paired_deltas,
    },
    "decision_rule":(
        "Attention is supported only if the frozen-local parallel branch improves "
        "untouched-test performance across paired seeds without damaging the local path."
    ),
    "claim_boundary":(
        "This tests attention as parallel residual evidence. It does not yet learn the "
        "Relation16 retrieval policy or define the 1,920 INFORMATION-byte semantics."
    ),
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"parallel_global_attention_residual_v8.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
