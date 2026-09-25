"""v10 — Top-K Global Attention Consensus.

v9 established useful complementary attention:
    local aggregate 1694/1800
    second-opinion aggregate 1715/1800
with 4 seed wins, 1 tie, 0 losses.

v10 changes ONLY the attention readout.

Instead of one nearest context, retrieve top K global contexts from TRAIN memory:
    K in {1,3,5,7}

Every context is classified by the SAME frozen local network. No labels participate
in retrieval or context voting.

For each query:
    local prediction pQ
    top-K context predictions pC_1..pC_K

Context consensus:
    vote[class] = count of top-K contexts predicting class
    pA = argmax vote
Tie-break:
    class whose nearest supporting context has lowest ring MEASURE energy.

Override local prediction only if:
    pA != pQ
    local top1-top2 score margin <= M_thr
    nearest context energy <= E_thr
    consensus count >= V_thr

K, M_thr, E_thr, V_thr are selected on VALIDATION only.
TEST is evaluated once after selection.

No attention weights, no Q+C collapse, no softmax, no test tuning.
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
KS=(1,3,5,7)

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

    def topk(self,Xq,maxk=7,training=False):
        ids=np.empty((len(Xq),maxk),dtype=np.int64)
        eng=np.empty((len(Xq),maxk),dtype=np.int64)

        for i,q in enumerate(Xq):
            cand=self.candidate_samples(q,exclude=(i if training else None))
            E=cdist_rows(q,self.X[cand])
            order=np.lexsort((cand,E))
            take=min(maxk,len(order))

            ids[i,:take]=cand[order[:take]]
            eng[i,:take]=E[order[:take]]

            if take<maxk:
                # Repeat last valid candidate only for shape stability.
                ids[i,take:]=ids[i,take-1]
                eng[i,take:]=eng[i,take-1]

        return ids,eng


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
                            ds[i,c]+=1; v+=1
                    ds[i,yi]-=v

                gW=a.T@ds
                gb=ds.sum(axis=0,dtype=np.int64)
                ga=ds@self.W.value.T
                gL=np.zeros(256,dtype=np.int64)
                np.add.at(gL,x.reshape(-1),(-ga).reshape(-1))

                self.W.step(gW); self.b.step(gb); self.L.step(gL)
                self.W.value=np.clip(self.W.value,-127,127)
                self.b.value=np.clip(self.b.value,-32767,32767)
                self.L.value&=255

            _,sv=self.scores(Xval)
            pv=sv.argmax(axis=1)
            vc=int((pv==yval).sum())
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


def quantile_candidates(x,qs):
    z=np.sort(np.asarray(x,dtype=np.int64))
    vals=[]
    for q in qs:
        idx=min(len(z)-1,(len(z)*q)//100)
        vals.append(int(z[idx]))
    return sorted(set(vals))


def context_predictions(net,index,ids):
    flat=index.X[ids.reshape(-1)]
    _,s=net.scores(flat)
    return s.argmax(axis=1).reshape(ids.shape).astype(np.int64)


def consensus(predK,energyK,K):
    """Return consensus class and winning vote count for first K contexts."""
    P=predK[:,:K]
    E=energyK[:,:K]
    out=np.empty(len(P),dtype=np.int64)
    votes=np.empty(len(P),dtype=np.int64)

    for i in range(len(P)):
        counts=np.bincount(P[i],minlength=NCLASS)
        vmax=int(counts.max())
        tied=np.where(counts==vmax)[0]

        if len(tied)==1:
            winner=int(tied[0])
        else:
            # Tie-break by nearest supporting ring-MEASURE energy.
            best=None
            for c in tied:
                ec=int(E[i][P[i]==c].min())
                key=(ec,int(c))
                if best is None or key<best[0]:
                    best=(key,int(c))
            winner=best[1]

        out[i]=winner
        votes[i]=vmax

    return out,votes


def apply_rule(pq,pa,margin,e1,votes,mt,et,vt):
    use=(
        (pq!=pa)
        & (margin<=int(mt))
        & (e1<=int(et))
        & (votes>=int(vt))
    )
    out=pq.copy()
    out[use]=pa[use]
    return out,use


# Fixed stratified 60/20/20.
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
ids_tr,Etr=index.topk(Xtr,maxk=max(KS),training=True)
ids_val,Eval=index.topk(Xval,maxk=max(KS),training=False)
ids_test,Etest=index.topk(Xtest,maxk=max(KS),training=False)

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
    pq_tr,mq_tr=score_stats(sq_tr)

    _,sq_val=net.scores(Xval)
    pq_val,mq_val=score_stats(sq_val)

    pred_tr=context_predictions(net,index,ids_tr)
    pred_val=context_predictions(net,index,ids_val)

    MTS=quantile_candidates(mq_tr,(10,25,50,75,90,100))
    ETS=quantile_candidates(Etr[:,0],(25,50,75,90,100))

    best=None

    for K in KS:
        pa_val,vv_val=consensus(pred_val,Eval,K)

        # Consensus thresholds are exact integer vote counts.
        VTS=range(1,K+1)

        for mt in MTS:
            for et in ETS:
                for vt in VTS:
                    p,use=apply_rule(
                        pq_val,pa_val,mq_val,Eval[:,0],vv_val,
                        mt,et,vt
                    )
                    ac=int((p==yval).sum())

                    # Prefer accuracy, then fewer overrides, then stronger consensus,
                    # then smaller K. All tie-breaks are validation-only.
                    key=(ac,-int(use.sum()),int(vt),-int(K))

                    if best is None or key>best[0]:
                        best=(key,(K,mt,et,vt,int(use.sum())))

    K,mt,et,vt,val_overrides=best[1]

    # TEST first used for model performance here.
    _,sq_test=net.scores(Xtest)
    pq_test,mq_test=score_stats(sq_test)
    pred_test=context_predictions(net,index,ids_test)
    pa_test,vv_test=consensus(pred_test,Etest,K)

    local_correct=int((pq_test==ytest).sum())
    pfinal,use=apply_rule(
        pq_test,pa_test,mq_test,Etest[:,0],vv_test,
        mt,et,vt
    )
    final_correct=int((pfinal==ytest).sum())

    local_wrong=(pq_test!=ytest)
    local_right=~local_wrong
    actual_rescues=int((local_wrong&(pfinal==ytest)).sum())
    actual_harms=int((local_right&(pfinal!=ytest)).sum())

    # Post-selection diagnostic: how often consensus itself is correct.
    consensus_correct=int((pa_test==ytest).sum())

    delta=final_correct-local_correct
    deltas.append(delta)

    per_seed.append({
        "seed":seed,
        "local_selection":sel,
        "rule":{
            "K":K,
            "local_margin_max":mt,
            "nearest_energy_max":et,
            "consensus_votes_min":vt,
            "validation_overrides":val_overrides,
            "validation_correct":best[0][0],
            "validation_total":len(yval),
        },
        "test":{
            "local_correct":local_correct,
            "topK_second_opinion_correct":final_correct,
            "consensus_alone_correct":consensus_correct,
            "total":len(ytest),
            "overrides":int(use.sum()),
            "actual_rescues":actual_rescues,
            "actual_harms":actual_harms,
            "delta":delta,
        },
    })

    print(
        "SEED",seed,
        "K",K,
        "LOCAL",local_correct,
        "TOPK",final_correct,
        "DELTA",delta,
        "OVERRIDE",int(use.sum()),
        "RESCUE",actual_rescues,
        "HARM",actual_harms,
        "VOTE>=",vt,
        flush=True,
    )

local_sum=sum(r["test"]["local_correct"] for r in per_seed)
final_sum=sum(r["test"]["topK_second_opinion_correct"] for r in per_seed)

report={
    "model":"v10-topK-global-attention-consensus",
    "architecture":{
        "local":"frozen local model on Q",
        "attention":"Relation16+MEASURE retrieves top K TRAIN contexts",
        "context_readout":"same frozen local model on every context",
        "fusion":"validation-selected consensus override",
        "extra_attention_weights":False,
        "query_context_collapsed":False,
    },
    "K_candidates":list(KS),
    "split":{
        "train":len(Xtr),
        "validation":len(Xval),
        "test_untouched":len(Xtest),
        "all_rule_selection_on_validation":True,
        "test_tuning":False,
    },
    "attention":{
        "labels_used_for_retrieval":False,
        "labels_used_for_consensus":False,
        "relation":"Relation16TransposePolicy",
        "measure":"sum circular Z256 distance",
    },
    "per_seed":per_seed,
    "aggregate":{
        "local_correct_sum":local_sum,
        "topK_second_opinion_correct_sum":final_sum,
        "paired_total_predictions":len(MODEL_SEEDS)*len(ytest),
        "delta_correct_sum":final_sum-local_sum,
        "seed_wins":sum(d>0 for d in deltas),
        "ties":sum(d==0 for d in deltas),
        "seed_losses":sum(d<0 for d in deltas),
        "deltas":deltas,
    },
    "decision_rule":(
        "Top-K consensus is supported only if validation-selected consensus improves "
        "untouched test more consistently than v9's single-context second opinion."
    ),
    "claim_boundary":(
        "v10 evaluates retrieval consensus, not learned attention relations or the "
        "1,920 INFORMATION-byte semantics."
    ),
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"topK_global_attention_consensus_v10.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
