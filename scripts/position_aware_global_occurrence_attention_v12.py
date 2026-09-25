"""v12 — Position-aware Global Occurrence Attention.

v11 diagnosed a semantic failure in candidate retrieval:

    "sample contains ANY query local state ANYWHERE"

saturated to all 1,077 TRAIN samples for EXACT, REL16, and FULL.

That discarded the global occurrence identity we had previously preserved.

v12 restores the occurrence pair:

    (local state p, local site h) -> global occurrence g = n*14464 + h

For the active 64-state probe each feature f has a fixed generator-7 manifold site h_f.

Policies
========
FULL:
    all TRAIN samples; diagnostic baseline.

SITE_EXACT:
    support(n,q) = number of active sites f for which X[n,f] == q[f]

SITE_REL16:
    support(n,q) = number of active sites f for which
                   X[n,f] in {q[f], tau16(q[f])}

Candidate ordering is label-free and lexicographic:
    1. larger positional support
    2. lower full 64-state ring MEASURE energy
    3. lower sample id

Only samples with support>0 enter SITE_* candidate sets.
If none survive, fall back to FULL and record it.

Attention readout is the successful v10 frozen-model top-K consensus.
K, local uncertainty, energy threshold, vote threshold, and minimum positional
support are selected on VALIDATION only.

No Q+C collapse. No attention weights. TEST untouched until final evaluation.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedShuffleSplit

from mprc_structural.global_attention import HV
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
POLICIES=("FULL","SITE_EXACT","SITE_REL16")

ORBIT=generator_orbit64()
SITE=np.asarray([int(ORBIT[t])*MAN_W+COL for t in range(FDIM)],dtype=np.int64)
assert len(set(map(int,SITE)))==FDIM


def tau16_vec(x):
    u=np.asarray(x,dtype=np.uint16)
    i=(u>>4)&15
    j=u&15
    return ((j<<4)|i).astype(np.uint8)


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


class PositionAwareIndex:
    def __init__(self,Xmem):
        self.X=np.asarray(Xmem,dtype=np.uint8)
        self.N=len(self.X)

    def support(self,q,policy):
        q=np.asarray(q,dtype=np.uint8)

        if policy=="FULL":
            return np.zeros(self.N,dtype=np.int64)

        exact=(self.X==q[None,:])
        if policy=="SITE_EXACT":
            return exact.sum(axis=1,dtype=np.int64)

        if policy=="SITE_REL16":
            tq=tau16_vec(q)
            related=(self.X==tq[None,:])
            return (exact|related).sum(axis=1,dtype=np.int64)

        raise KeyError(policy)

    def topk(self,Xq,policy,maxk=7,training=False):
        ids=np.empty((len(Xq),maxk),dtype=np.int64)
        eng=np.empty((len(Xq),maxk),dtype=np.int64)
        sup=np.empty((len(Xq),maxk),dtype=np.int64)
        cand_count=np.empty(len(Xq),dtype=np.int64)
        fallback=np.zeros(len(Xq),dtype=np.bool_)

        all_ids=np.arange(self.N,dtype=np.int64)

        for i,q in enumerate(Xq):
            support=self.support(q,policy)

            if policy=="FULL":
                cand=all_ids
            else:
                cand=np.where(support>0)[0].astype(np.int64)
                if training:
                    cand=cand[cand!=i]
                if len(cand)==0:
                    fallback[i]=True
                    cand=all_ids
                    if training:
                        cand=cand[cand!=i]

            if policy=="FULL" and training:
                cand=cand[cand!=i]

            cand_count[i]=len(cand)
            E=cdist_rows(q,self.X[cand])
            S=support[cand]

            # np.lexsort last key is primary:
            # primary -support, then energy, then sample id.
            order=np.lexsort((cand,E,-S))
            take=min(maxk,len(order))

            picked=cand[order[:take]]
            ids[i,:take]=picked
            eng[i,:take]=E[order[:take]]
            sup[i,:take]=support[picked]

            if take<maxk:
                ids[i,take:]=ids[i,take-1]
                eng[i,take:]=eng[i,take-1]
                sup[i,take:]=sup[i,take-1]

        return ids,eng,sup,cand_count,fallback


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
        return {"L":self.L.value.copy(),"W":self.W.value.copy(),"b":self.b.value.copy()}

    def restore(self,c):
        self.L.value=c["L"].copy()
        self.W.value=c["W"].copy()
        self.b.value=c["b"].copy()

    def train_select_validation(self,Xtr,ytr,Xval,yval,orders):
        best=(-1,0); best_ck=None
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

            _,sv=self.scores(Xval)
            pv=sv.argmax(axis=1)
            ac=int((pv==yval).sum())
            if ac>best[0]:
                best=(ac,ep)
                best_ck=self.ckpt()

        self.restore(best_ck)
        return {"selected_validation":[best[0],len(yval)],"selected_epoch":best[1]}


def score_stats(s):
    order=np.argsort(s,axis=1)
    p1=order[:,-1]; p2=order[:,-2]
    r=np.arange(len(s))
    m=s[r,p1]-s[r,p2]
    return p1.astype(np.int64),m.astype(np.int64)


def quantiles(x,qs):
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


def consensus(pred,E,K):
    P=pred[:,:K]; Z=E[:,:K]
    out=np.empty(len(P),dtype=np.int64)
    votes=np.empty(len(P),dtype=np.int64)

    for i in range(len(P)):
        cnt=np.bincount(P[i],minlength=NCLASS)
        vmax=int(cnt.max())
        tied=np.where(cnt==vmax)[0]
        if len(tied)==1:
            w=int(tied[0])
        else:
            best=None
            for c in tied:
                e=int(Z[i][P[i]==c].min())
                key=(e,int(c))
                if best is None or key<best[0]:
                    best=(key,int(c))
            w=best[1]
        out[i]=w; votes[i]=vmax

    return out,votes


def apply_rule(pq,pa,margin,e1,s1,votes,mt,et,st,vt):
    use=(
        (pq!=pa)
        & (margin<=int(mt))
        & (e1<=int(et))
        & (s1>=int(st))
        & (votes>=int(vt))
    )
    out=pq.copy()
    out[use]=pa[use]
    return out,use


def choose_rule(pqv,mqv,predv,Ev,Sv,yv,mqtr,Etr,Str):
    MTS=quantiles(mqtr,(10,25,50,75,90,100))
    ETS=quantiles(Etr[:,0],(25,50,75,90,100))
    STS=sorted(set([0]+quantiles(Str[:,0],(0,25,50,75,90,100))))

    best=None
    for K in KS:
        pa,vv=consensus(predv,Ev,K)
        for mt in MTS:
            for et in ETS:
                for st in STS:
                    for vt in range(1,K+1):
                        p,use=apply_rule(pqv,pa,mqv,Ev[:,0],Sv[:,0],vv,mt,et,st,vt)
                        ac=int((p==yv).sum())
                        key=(ac,-int(use.sum()),int(st),int(vt),-int(K))
                        if best is None or key>best[0]:
                            best=(key,(K,mt,et,st,vt,int(use.sum())))
    return best


# Fixed 60/20/20 split.
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

index=PositionAwareIndex(Xtr)

retrieval={}
for policy in POLICIES:
    retrieval[policy]={
        "train":index.topk(Xtr,policy,maxk=max(KS),training=True),
        "val":index.topk(Xva,policy,maxk=max(KS),training=False),
        "test":index.topk(Xte,policy,maxk=max(KS),training=False),
    }

per_seed=[]
agg={p:{"local_sum":0,"attn_sum":0,"deltas":[]} for p in POLICIES}

for seed in MODEL_SEEDS:
    rng=np.random.default_rng(seed)
    W0=rng.integers(-2,3,size=(FDIM,NCLASS),dtype=np.int64)
    b0=np.zeros(NCLASS,dtype=np.int64)
    L0=np.arange(256,dtype=np.int64)
    rr=np.random.default_rng(seed+100000)
    orders=[rr.permutation(len(Xtr)) for _ in range(EPOCHS)]

    net=LocalNet(W0,b0,L0)
    local_sel=net.train_select_validation(Xtr,ytr,Xva,yva,orders)

    _,str_=net.scores(Xtr); pqtr,mqtr=score_stats(str_)
    _,sva=net.scores(Xva); pqva,mqva=score_stats(sva)
    _,ste=net.scores(Xte); pqte,mqte=score_stats(ste)
    local_correct=int((pqte==yte).sum())

    row={"seed":seed,"local_selection":local_sel,"local_test_correct":local_correct,"policies":{}}

    for policy in POLICIES:
        idtr,Etr,Str,Ctr,Ftr=retrieval[policy]["train"]
        idva,Eva,Sva,Cva,Fva=retrieval[policy]["val"]
        idte,Ete,Ste,Cte,Fte=retrieval[policy]["test"]

        predva=context_predictions(net,index,idva)
        best=choose_rule(pqva,mqva,predva,Eva,Sva,yva,mqtr,Etr,Str)
        K,mt,et,st,vt,val_overrides=best[1]

        predte=context_predictions(net,index,idte)
        pate,vte=consensus(predte,Ete,K)
        final,use=apply_rule(pqte,pate,mqte,Ete[:,0],Ste[:,0],vte,mt,et,st,vt)
        final_correct=int((final==yte).sum())

        wrong=pqte!=yte
        right=~wrong
        rescue=int((wrong&(final==yte)).sum())
        harm=int((right&(final!=yte)).sum())
        delta=final_correct-local_correct

        row["policies"][policy]={
            "rule":{
                "K":K,
                "margin_max":mt,
                "energy_max":et,
                "support_min":st,
                "votes_min":vt,
                "validation_overrides":val_overrides,
            },
            "test_correct":final_correct,
            "delta":delta,
            "overrides":int(use.sum()),
            "rescues":rescue,
            "harms":harm,
            "candidate_count_mean":float(Cte.mean()),
            "candidate_count_min":int(Cte.min()),
            "candidate_count_max":int(Cte.max()),
            "fallback_count":int(Fte.sum()),
            "top1_support_mean":float(Ste[:,0].mean()),
        }

        agg[policy]["local_sum"]+=local_correct
        agg[policy]["attn_sum"]+=final_correct
        agg[policy]["deltas"].append(delta)

        print(
            "SEED",seed,policy,
            "LOCAL",local_correct,
            "ATTN",final_correct,
            "DELTA",delta,
            "K",K,
            "SUP>=",st,
            "CAND_MEAN",float(Cte.mean()),
            "TOP1_SUP",float(Ste[:,0].mean()),
            flush=True,
        )

    per_seed.append(row)

for p in POLICIES:
    d=agg[p]["deltas"]
    agg[p]["delta_sum"]=agg[p]["attn_sum"]-agg[p]["local_sum"]
    agg[p]["wins"]=sum(x>0 for x in d)
    agg[p]["ties"]=sum(x==0 for x in d)
    agg[p]["losses"]=sum(x<0 for x in d)

report={
    "model":"v12-position-aware-global-occurrence-attention",
    "diagnosis_from_v11":"ANY-state sample union saturated all 1077 TRAIN samples; local/global occurrence was lost",
    "global_address":"g=n*14464+h_f for each active site f",
    "policies":{
        "FULL":"all train samples",
        "SITE_EXACT":"same active site, exact Z256 state",
        "SITE_REL16":"same active site, state p or candidate tau16(p)",
    },
    "ranking":"higher positional support, then lower 64-state ring MEASURE, then sample id",
    "attention_readout":"v10 top-K frozen-model consensus with validation-selected override",
    "labels_used_for_retrieval":False,
    "split":{"train":len(Xtr),"validation":len(Xva),"test":len(Xte),"test_tuning":False},
    "per_seed":per_seed,
    "aggregate":agg,
    "claim_boundary":(
        "v12 restores positional occurrence structure. SITE_REL16 remains an empirical "
        "candidate relation, not a frozen theorem. This active 64-state mapping is a "
        "probe of local/global addressing, not a claim that the digits grid is the MPRC manifold."
    ),
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"position_aware_global_occurrence_attention_v12.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
