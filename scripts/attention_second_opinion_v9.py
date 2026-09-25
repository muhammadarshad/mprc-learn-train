"""v9 — Attention as a label-free second opinion.

v8 showed the trained residual attention branch was almost neutral. This probe
asks a simpler question: does retrieved context provide complementary decisions
exactly when the local model is uncertain?

For each input:
    local prediction  yQ = argmax S(Q)
    context prediction yC = argmax S(C)
where C is selected label-free by Relation16Transpose + ring MEASURE over TRAIN.

Attention may override yQ only when:
    1) yQ != yC
    2) local top1-top2 margin <= M_thr
    3) nearest ring energy E1 <= E_thr
    4) retrieval separation (E2-E1) >= G_thr

Threshold candidates are derived from TRAIN diagnostics and selected on
VALIDATION only. TEST is evaluated once after rule selection.

No extra attention weights, no Q+C collapse, no softmax, no test tuning.
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
EPOCHS=60
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

    def step(self,grad):
        self.residual+=np.asarray(grad,dtype=np.int64)
        k=trunc_div_array(self.residual,DEN)
        self.residual-=k*DEN
        self.value-=k
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
            b0=int(order[0])
            b1=int(order[1]) if len(order)>1 else b0
            selected[i]=int(cand[b0])
            C[i]=self.X[selected[i]]
            e1[i]=int(E[b0])
            e2[i]=int(E[b1])
        return C,selected,e1,e2


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

    def checkpoint(self):
        return {
            "L":self.L.value.copy(),
            "W":self.W.value.copy(),
            "b":self.b.value.copy(),
        }

    def restore(self,c):
        self.L.value=c["L"].copy()
        self.W.value=c["W"].copy()
        self.b.value=c["b"].copy()

    def train_select_validation(self,Xtr,ytr,Xval,yval,orders):
        best=(-1,0)
        best_ck=None

        for ep,order in enumerate(orders,1):
            for st in range(0,len(order),BATCH):
                ids=order[st:st+BATCH]
                x=Xtr[ids]; yy=ytr[ids]
                a,s=self.scores(x)
                ds=np.zeros_like(s,dtype=np.int64)

                for i in range(len(ids)):
                    yi=int(yy[i]); sy=int(s[i,yi]); v=0
                    for c in range(NCLASS):
                        if c==yi:
                            continue
                        if int(s[i,c])+MARGIN>sy:
                            ds[i,c]+=1
                            v+=1
                    ds[i,yi]-=v

                gW=a.T@ds
                gb=ds.sum(axis=0,dtype=np.int64)
                ga=ds@self.W.value.T
                gL=np.zeros(256,dtype=np.int64)
                np.add.at(gL,x.reshape(-1),(-ga).reshape(-1))

                self.W.step(gW)
                self.b.step(gb)
                self.L.step(gL)
                self.W.value=np.clip(self.W.value,-127,127)
                self.b.value=np.clip(self.b.value,-32767,32767)
                self.L.value&=255

            _,sv=self.scores(Xval)
            pred=sv.argmax(axis=1)
            vc=int((pred==yval).sum())
            if vc>best[0]:
                best=(vc,ep)
                best_ck=self.checkpoint()

        self.restore(best_ck)
        return {"selected_validation":[best[0],len(yval)],"selected_epoch":best[1]}


def score_stats(scores):
    order=np.argsort(scores,axis=1)
    top=order[:,-1]
    second=order[:,-2]
    rows=np.arange(len(scores))
    margin=scores[rows,top]-scores[rows,second]
    return top.astype(np.int64),margin.astype(np.int64)


def quantile_candidates(x,qs=(10,25,50,75,90,100)):
    z=np.sort(np.asarray(x,dtype=np.int64))
    out=[]
    for q in qs:
        idx=min(len(z)-1,(len(z)*q)//100)
        out.append(int(z[idx]))
    return sorted(set(out))


def apply_rule(predQ,predC,local_margin,e1,gap,mt,et,gt):
    use=(
        (predQ!=predC)
        & (local_margin<=int(mt))
        & (e1<=int(et))
        & (gap>=int(gt))
    )
    out=predQ.copy()
    out[use]=predC[use]
    return out,use


# Fixed 60/20/20 split.
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

index=ActiveGlobalTranspose(Xtr)
Ctr,seltr,Etr,Etr2=index.select(Xtr,training=True)
Cval,selval,Eval,Eval2=index.select(Xval,training=False)
Ctest,seltest,Etest,Etest2=index.select(Xtest,training=False)

per_seed=[]
deltas=[]

for seed in MODEL_SEEDS:
    rng=np.random.default_rng(seed)
    W0=rng.integers(-2,3,size=(FDIM,NCLASS),dtype=np.int64)
    b0=np.zeros(NCLASS,dtype=np.int64)
    L0=np.arange(256,dtype=np.int64)

    order_rng=np.random.default_rng(seed+100000)
    orders=[order_rng.permutation(len(Xtr)) for _ in range(EPOCHS)]

    net=LocalNet(W0,b0,L0)
    sel=net.train_select_validation(Xtr,ytr,Xval,yval,orders)

    _,sq_tr=net.scores(Xtr)
    _,sc_tr=net.scores(Ctr)
    pq_tr,mq_tr=score_stats(sq_tr)
    pc_tr,mc_tr=score_stats(sc_tr)

    _,sq_val=net.scores(Xval)
    _,sc_val=net.scores(Cval)
    pq_val,mq_val=score_stats(sq_val)
    pc_val,mc_val=score_stats(sc_val)

    gap_tr=Etr2-Etr
    gap_val=Eval2-Eval

    MTS=quantile_candidates(mq_tr,(10,25,50,75,90,100))
    ETS=quantile_candidates(Etr,(25,50,75,90,100))
    GTS=quantile_candidates(gap_tr,(0,10,25,50,75,90))

    best=(-1,None)
    for mt in MTS:
        for et in ETS:
            for gt in GTS:
                p,use=apply_rule(pq_val,pc_val,mq_val,Eval,gap_val,mt,et,gt)
                ac=int((p==yval).sum())
                # Tie-break: fewer overrides, then tighter local-margin threshold.
                key=(ac,-int(use.sum()),-int(mt))
                if best[1] is None or key>best[0]:
                    best=(key,(mt,et,gt,int(use.sum())))

    mt,et,gt,val_overrides=best[1]

    # TEST first used for model performance here.
    _,sq_test=net.scores(Xtest)
    _,sc_test=net.scores(Ctest)
    pq_test,mq_test=score_stats(sq_test)
    pc_test,mc_test=score_stats(sc_test)
    gap_test=Etest2-Etest

    local_correct=int((pq_test==ytest).sum())
    pfinal,use=apply_rule(pq_test,pc_test,mq_test,Etest,gap_test,mt,et,gt)
    final_correct=int((pfinal==ytest).sum())

    # Post-selection diagnostics on TEST.
    local_wrong=(pq_test!=ytest)
    local_right=~local_wrong
    ctx_correct=(pc_test==ytest)
    disagreement=(pq_test!=pc_test)

    rescue_possible=int((local_wrong&ctx_correct&disagreement).sum())
    harm_possible=int((local_right&(~ctx_correct)&disagreement).sum())
    actual_rescues=int((local_wrong&(pfinal==ytest)).sum())
    actual_harms=int((local_right&(pfinal!=ytest)).sum())

    per_seed.append({
        "seed":seed,
        "local_selection":sel,
        "rule":{
            "local_margin_max":mt,
            "measure_energy_max":et,
            "measure_gap_min":gt,
            "validation_overrides":val_overrides,
            "validation_correct":best[0][0],
            "validation_total":len(yval),
        },
        "test":{
            "local_correct":local_correct,
            "second_opinion_correct":final_correct,
            "total":len(ytest),
            "overrides":int(use.sum()),
            "delta":final_correct-local_correct,
            "rescue_possible":rescue_possible,
            "harm_possible":harm_possible,
            "actual_rescues":actual_rescues,
            "actual_harms":actual_harms,
        },
    })

    deltas.append(final_correct-local_correct)

    print(
        "SEED",seed,
        "LOCAL",local_correct,
        "SECOND",final_correct,
        "DELTA",final_correct-local_correct,
        "OVERRIDE",int(use.sum()),
        "RESCUE",actual_rescues,
        "HARM",actual_harms,
        "RULE",(mt,et,gt),
        flush=True,
    )

local_sum=sum(r["test"]["local_correct"] for r in per_seed)
second_sum=sum(r["test"]["second_opinion_correct"] for r in per_seed)

# Retrieval-label diagnostic only after rule selection.
neighbor_label_agreement=int((ytr[seltest]==ytest).sum())

report={
    "model":"v9-attention-second-opinion",
    "architecture":{
        "local_prediction":"argmax S(Q)",
        "context_prediction":"argmax S(C) using same frozen local model",
        "fusion":"override only on validation-selected uncertainty/retrieval-confidence rule",
        "extra_attention_weights":False,
        "query_context_collapsed":False,
    },
    "signals":{
        "local_uncertainty":"top1-top2 integer score margin",
        "retrieval_strength":"nearest ring MEASURE energy E1",
        "retrieval_separation":"E2-E1",
    },
    "split":{
        "train":len(Xtr),
        "validation":len(Xval),
        "test_untouched":len(Xtest),
        "rule_selected_on_validation":True,
        "test_tuning":False,
    },
    "attention":{
        "memory":"TRAIN only",
        "labels_used_for_retrieval":False,
        "policy":"Relation16TransposePolicy",
        "test_neighbor_label_agreement_diagnostic":[neighbor_label_agreement,len(ytest)],
    },
    "per_seed":per_seed,
    "aggregate":{
        "local_correct_sum":local_sum,
        "second_opinion_correct_sum":second_sum,
        "paired_total_predictions":len(MODEL_SEEDS)*len(ytest),
        "delta_correct_sum":second_sum-local_sum,
        "seed_wins":sum(d>0 for d in deltas),
        "ties":sum(d==0 for d in deltas),
        "seed_losses":sum(d<0 for d in deltas),
        "deltas":deltas,
    },
    "decision_rule":(
        "If validation-selected second-opinion overrides do not improve untouched test "
        "consistently, the current retrieval signal is largely redundant with local evidence."
    ),
    "claim_boundary":(
        "v9 tests complementary decision value of current Relation16+MEASURE retrieval. "
        "It does not define INFORMATION semantics or learn the retrieval relation."
    ),
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"attention_second_opinion_v9.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
