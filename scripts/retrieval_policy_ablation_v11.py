"""v11 — Retrieval policy ablation for successful top-K attention.

v10 established a leakage-free gain:
    local 1694/1800
    top-K attention 1716/1800
    5 seed wins, 0 losses.

Now isolate WHERE that gain comes from.

Policies
========
EXACT:
    candidate samples must contain exact local state p.

REL16:
    candidate samples may contain p or tau16(p).
    tau16 is the current candidate 16x16 transpose relation.

FULL:
    no local-state filter; ring MEASURE ranks all TRAIN samples.

All policies use the SAME:
- frozen local model
- TRAIN memory only
- ring MEASURE
- K candidates {1,3,5,7}
- local-uncertainty + nearest-energy + consensus threshold rule
- validation-only rule selection
- untouched TEST evaluation once per predeclared policy

The purpose is not to pick a winner from TEST. It is a predeclared ablation asking
whether Relation16 candidate structure adds value beyond exact lookup or full scan.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit

from mprc_structural.global_attention import HV, ExactStatePolicy, Relation16TransposePolicy
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
POLICIES=("EXACT","REL16","FULL")

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


class RetrievalIndex:
    def __init__(self,Xmem):
        self.X=np.asarray(Xmem,dtype=np.uint8)
        self.N=len(self.X)
        buckets=[[] for _ in range(256)]
        for n,row in enumerate(self.X):
            base=n*HV
            for f,p0 in enumerate(row):
                buckets[int(p0)].append((base+int(SITE[f]),n,f))
        self.buckets=buckets
        self.exact=ExactStatePolicy()
        self.rel16=Relation16TransposePolicy()

    def candidate_samples(self,q,policy,exclude=None):
        if policy=="FULL":
            out=np.arange(self.N,dtype=np.int64)
            if exclude is not None:
                out=out[out!=int(exclude)]
            return out

        relation=self.exact if policy=="EXACT" else self.rel16
        seen=set()

        for p0 in q:
            for p in relation.states(int(p0)):
                for g,n,f in self.buckets[p]:
                    if exclude is not None and n==exclude:
                        continue
                    seen.add(n)

        if not seen:
            seen=set(range(self.N))
            if exclude is not None:
                seen.discard(exclude)

        return np.asarray(sorted(seen),dtype=np.int64)

    def topk(self,Xq,policy,maxk=7,training=False):
        ids=np.empty((len(Xq),maxk),dtype=np.int64)
        eng=np.empty((len(Xq),maxk),dtype=np.int64)
        candidate_counts=np.empty(len(Xq),dtype=np.int64)

        for i,q in enumerate(Xq):
            cand=self.candidate_samples(q,policy,exclude=(i if training else None))
            candidate_counts[i]=len(cand)
            E=cdist_rows(q,self.X[cand])
            order=np.lexsort((cand,E))
            take=min(maxk,len(order))
            ids[i,:take]=cand[order[:take]]
            eng[i,:take]=E[order[:take]]
            if take<maxk:
                ids[i,take:]=ids[i,take-1]
                eng[i,take:]=eng[i,take-1]

        return ids,eng,candidate_counts


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

                self.W.step(gW); self.b.step(gb); self.L.step(gL)
                self.W.value=np.clip(self.W.value,-127,127)
                self.b.value=np.clip(self.b.value,-32767,32767)
                self.L.value&=255

            _,sv=self.scores(Xval)
            pv=sv.argmax(axis=1)
            vc=int((pv==yval).sum())
            if vc>best[0]:
                best=(vc,ep)
                best_ck=self.ckpt()

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


def choose_rule(pq_val,mq_val,pred_val,E_val,yval,mq_train,E_train):
    MTS=quantile_candidates(mq_train,(10,25,50,75,90,100))
    ETS=quantile_candidates(E_train[:,0],(25,50,75,90,100))
    best=None

    for K in KS:
        pa,vv=consensus(pred_val,E_val,K)
        for mt in MTS:
            for et in ETS:
                for vt in range(1,K+1):
                    p,use=apply_rule(pq_val,pa,mq_val,E_val[:,0],vv,mt,et,vt)
                    ac=int((p==yval).sum())
                    key=(ac,-int(use.sum()),int(vt),-int(K))
                    if best is None or key>best[0]:
                        best=(key,(K,mt,et,vt,int(use.sum())))
    return best


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

index=RetrievalIndex(Xtr)

# Retrieval is independent of model seed, so build once per policy.
retrieval={}
for policy in POLICIES:
    itr,Etr,Ctr=index.topk(Xtr,policy,maxk=max(KS),training=True)
    iva,Eva,Cva=index.topk(Xval,policy,maxk=max(KS),training=False)
    ite,Ete,Cte=index.topk(Xtest,policy,maxk=max(KS),training=False)
    retrieval[policy]={
        "train":(itr,Etr,Ctr),
        "val":(iva,Eva,Cva),
        "test":(ite,Ete,Cte),
    }

per_seed=[]
aggregate={p:{"local_sum":0,"attention_sum":0,"deltas":[]} for p in POLICIES}

for seed in MODEL_SEEDS:
    rng=np.random.default_rng(seed)
    W0=rng.integers(-2,3,size=(FDIM,NCLASS),dtype=np.int64)
    b0=np.zeros(NCLASS,dtype=np.int64)
    L0=np.arange(256,dtype=np.int64)
    order_rng=np.random.default_rng(seed+100000)
    orders=[order_rng.permutation(len(Xtr)) for _ in range(EPOCHS)]

    net=LocalNet(W0,b0,L0)
    local_sel=net.train_select_validation(Xtr,ytr,Xval,yval,orders)

    _,sq_tr=net.scores(Xtr)
    pq_tr,mq_tr=score_stats(sq_tr)
    _,sq_val=net.scores(Xval)
    pq_val,mq_val=score_stats(sq_val)

    _,sq_test=net.scores(Xtest)
    pq_test,mq_test=score_stats(sq_test)
    local_correct=int((pq_test==ytest).sum())

    seed_row={"seed":seed,"local_selection":local_sel,"local_test_correct":local_correct,"policies":{}}

    for policy in POLICIES:
        ids_tr,Etr,Ctr=retrieval[policy]["train"]
        ids_val,Eval,Cval=retrieval[policy]["val"]
        ids_test,Etest,Ctest=retrieval[policy]["test"]

        pred_tr=context_predictions(net,index,ids_tr)
        pred_val=context_predictions(net,index,ids_val)

        best=choose_rule(pq_val,mq_val,pred_val,Eval,yval,mq_tr,Etr)
        K,mt,et,vt,val_overrides=best[1]

        pred_test=context_predictions(net,index,ids_test)
        pa_test,vv_test=consensus(pred_test,Etest,K)
        pfinal,use=apply_rule(pq_test,pa_test,mq_test,Etest[:,0],vv_test,mt,et,vt)
        final_correct=int((pfinal==ytest).sum())

        local_wrong=pq_test!=ytest
        local_right=~local_wrong
        rescues=int((local_wrong&(pfinal==ytest)).sum())
        harms=int((local_right&(pfinal!=ytest)).sum())

        delta=final_correct-local_correct
        candidate_mean=float(Ctest.mean())

        seed_row["policies"][policy]={
            "rule":{"K":K,"margin_max":mt,"energy_max":et,"votes_min":vt,"validation_overrides":val_overrides},
            "test_correct":final_correct,
            "delta":delta,
            "overrides":int(use.sum()),
            "rescues":rescues,
            "harms":harms,
            "mean_candidate_samples":candidate_mean,
        }

        aggregate[policy]["local_sum"]+=local_correct
        aggregate[policy]["attention_sum"]+=final_correct
        aggregate[policy]["deltas"].append(delta)

        print(
            "SEED",seed,policy,
            "LOCAL",local_correct,
            "ATTN",final_correct,
            "DELTA",delta,
            "K",K,
            "CAND_MEAN",candidate_mean,
            flush=True,
        )

    per_seed.append(seed_row)

for policy in POLICIES:
    a=aggregate[policy]
    d=a["deltas"]
    a["delta_sum"]=a["attention_sum"]-a["local_sum"]
    a["wins"]=sum(x>0 for x in d)
    a["ties"]=sum(x==0 for x in d)
    a["losses"]=sum(x<0 for x in d)

report={
    "model":"v11-retrieval-policy-ablation",
    "predeclared_policies":{
        "EXACT":"candidate samples contain exact local state p",
        "REL16":"candidate samples contain p or tau16(p)",
        "FULL":"all TRAIN samples enter ring MEASURE ranking",
    },
    "shared_attention":"top-K frozen-model context consensus + validation-selected uncertainty/confidence override",
    "split":{"train":len(Xtr),"validation":len(Xval),"test":len(Xtest),"test_tuning":False},
    "per_seed":per_seed,
    "aggregate":aggregate,
    "claim_boundary":(
        "This is a predeclared retrieval-policy ablation. REL16 remains a candidate "
        "16x16 relation, not a frozen extension of the paper's 3x3 Transpose theorem. "
        "FULL is a diagnostic baseline, not an efficient implementation recommendation."
    ),
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"retrieval_policy_ablation_v11.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
