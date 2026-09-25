"""v13 — Generator-displacement Global Attention.

v12 benchmark:
    local       1694/1800 = 94.11%
    SITE_EXACT  1719/1800 = 95.50%  (+25)
    SITE_REL16  1716/1800 = 95.33%

The useful new signal was SAME-POSITION exact-state support. Relation16 value
expansion did not improve it.

v13 changes the POSITION relation only.

The 64 active states are already ordered by the GEN=7 orbit. Therefore one
logical feature displacement corresponds to one generator step in physical
manifold row position.

For each query q and training sample x, for allowed displacement d:

    support_d(q,x) = count_f [ q[f] == x[(f+d) mod 64] ]

Select the displacement d* with maximum support.
Tie-break by:
    1) lower ring MEASURE after aligning x back by -d*
    2) smaller |d*|
    3) deterministic signed d order

The aligned context:
    C[f] = x[(f+d*) mod 64]

is classified by the same frozen local model.

Allowed displacement radii are selected on VALIDATION:
    R in {0,1,2,4,7}

R=0 exactly reproduces v12 SITE_EXACT semantics.

No Relation16 value expansion, no Q+C collapse, no attention weights.
TEST remains untouched until validation selects the rule.
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
RADII=(0,1,2,4,7)
KS=(1,3,5,7)


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

    def train_select_validation(self,Xtr,ytr,Xva,yva,orders):
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

            _,sv=self.scores(Xva)
            pv=sv.argmax(axis=1)
            ac=int((pv==yva).sum())
            if ac>best[0]:
                best=(ac,ep)
                best_ck=self.ckpt()

        self.restore(best_ck)
        return {"selected_validation":[best[0],len(yva)],"selected_epoch":best[1]}


def score_stats(s):
    order=np.argsort(s,axis=1)
    p1=order[:,-1];p2=order[:,-2]
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


def displacement_set(R):
    if R==0:
        return (0,)
    return tuple(range(-R,R+1))


class DisplacementRetrieval:
    def __init__(self,Xmem):
        self.X=np.asarray(Xmem,dtype=np.uint8)
        self.N=len(self.X)

    def rank(self,q,R,exclude=None,maxk=7):
        ds=displacement_set(R)
        ids=np.arange(self.N,dtype=np.int64)
        if exclude is not None:
            ids=ids[ids!=int(exclude)]
        X=self.X[ids]

        best_support=np.full(len(ids),-1,dtype=np.int64)
        best_energy=np.full(len(ids),np.iinfo(np.int64).max,dtype=np.int64)
        best_d=np.zeros(len(ids),dtype=np.int64)

        for d in ds:
            aligned=np.roll(X,-int(d),axis=1)
            sup=(aligned==q[None,:]).sum(axis=1,dtype=np.int64)
            E=cdist_rows(q,aligned)

            better=(
                (sup>best_support)
                | ((sup==best_support)&(E<best_energy))
                | ((sup==best_support)&(E==best_energy)&(abs(d)<np.abs(best_d)))
                | ((sup==best_support)&(E==best_energy)&(abs(d)==np.abs(best_d))&(d<best_d))
            )

            best_support[better]=sup[better]
            best_energy[better]=E[better]
            best_d[better]=int(d)

        order=np.lexsort((ids,np.abs(best_d),best_energy,-best_support))
        take=min(maxk,len(order))
        o=order[:take]

        sel_ids=ids[o]
        sel_sup=best_support[o]
        sel_E=best_energy[o]
        sel_d=best_d[o]

        C=np.empty((take,FDIM),dtype=np.uint8)
        for j,(sid,d) in enumerate(zip(sel_ids,sel_d)):
            C[j]=np.roll(self.X[int(sid)],-int(d))

        return sel_ids,sel_E,sel_sup,sel_d,C

    def batch(self,Xq,R,training=False,maxk=7):
        n=len(Xq)
        ids=np.empty((n,maxk),dtype=np.int64)
        E=np.empty((n,maxk),dtype=np.int64)
        S=np.empty((n,maxk),dtype=np.int64)
        D=np.empty((n,maxk),dtype=np.int64)
        C=np.empty((n,maxk,FDIM),dtype=np.uint8)

        for i,q in enumerate(Xq):
            a,b,c,d,e=self.rank(q,R,exclude=(i if training else None),maxk=maxk)
            take=len(a)
            ids[i,:take]=a;E[i,:take]=b;S[i,:take]=c;D[i,:take]=d;C[i,:take]=e
            if take<maxk:
                ids[i,take:]=ids[i,take-1]
                E[i,take:]=E[i,take-1]
                S[i,take:]=S[i,take-1]
                D[i,take:]=D[i,take-1]
                C[i,take:]=C[i,take-1]
        return ids,E,S,D,C


def context_predictions(net,C):
    flat=C.reshape(-1,FDIM)
    _,s=net.scores(flat)
    return s.argmax(axis=1).reshape(C.shape[:2]).astype(np.int64)


def consensus(pred,E,K):
    P=pred[:,:K];Z=E[:,:K]
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

        out[i]=w;votes[i]=vmax
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


def choose_rule(pqv,mqv,retrieval_by_R,yv,mqtr,retrieval_train_by_R):
    MTS=quantiles(mqtr,(10,25,50,75,90,100))
    best=None

    for R in RADII:
        Etr,Str,Dtr,Ptr=retrieval_train_by_R[R]
        Eva,Sva,Dva,Pva=retrieval_by_R[R]

        ETS=quantiles(Etr[:,0],(25,50,75,90,100))
        STS=quantiles(Str[:,0],(0,25,50,75,90,100))

        for K in KS:
            pa,vv=consensus(Pva,Eva,K)

            for mt in MTS:
                for et in ETS:
                    for st in STS:
                        for vt in range(1,K+1):
                            p,use=apply_rule(
                                pqv,pa,mqv,Eva[:,0],Sva[:,0],vv,
                                mt,et,st,vt
                            )
                            ac=int((p==yv).sum())

                            # Prefer accuracy, fewer overrides, smaller radius,
                            # stronger support, stronger vote, smaller K.
                            key=(ac,-int(use.sum()),-int(R),int(st),int(vt),-int(K))

                            if best is None or key>best[0]:
                                best=(key,(R,K,mt,et,st,vt,int(use.sum())))
    return best


# Fixed 60/20/20 split.
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

retr=DisplacementRetrieval(Xtr)

# Retrieval is model-independent.
raw={}
for R in RADII:
    raw[R]={
        "train":retr.batch(Xtr,R,training=True,maxk=max(KS)),
        "val":retr.batch(Xva,R,training=False,maxk=max(KS)),
        "test":retr.batch(Xte,R,training=False,maxk=max(KS)),
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
    local_sel=net.train_select_validation(Xtr,ytr,Xva,yva,orders)

    _,str_=net.scores(Xtr);pqtr,mqtr=score_stats(str_)
    _,sva=net.scores(Xva);pqva,mqva=score_stats(sva)
    _,ste=net.scores(Xte);pqte,mqte=score_stats(ste)
    local_correct=int((pqte==yte).sum())

    train_prepared={}
    val_prepared={}

    for R in RADII:
        _,Etr,Str,Dtr,Ctr=raw[R]["train"]
        _,Eva,Sva,Dva,Cva=raw[R]["val"]
        Ptr=context_predictions(net,Ctr)
        Pva=context_predictions(net,Cva)
        train_prepared[R]=(Etr,Str,Dtr,Ptr)
        val_prepared[R]=(Eva,Sva,Dva,Pva)

    best=choose_rule(pqva,mqva,val_prepared,yva,mqtr,train_prepared)
    R,K,mt,et,st,vt,val_overrides=best[1]

    _,Ete,Ste,Dte,Cte=raw[R]["test"]
    Pte=context_predictions(net,Cte)
    pate,vte=consensus(Pte,Ete,K)

    final,use=apply_rule(
        pqte,pate,mqte,Ete[:,0],Ste[:,0],vte,
        mt,et,st,vt
    )

    final_correct=int((final==yte).sum())
    delta=final_correct-local_correct
    wrong=pqte!=yte
    right=~wrong
    rescue=int((wrong&(final==yte)).sum())
    harm=int((right&(final!=yte)).sum())

    # Post-selection displacement diagnostics.
    used_d=Dte[:,0]
    displacement_hist={str(d):int((used_d==d).sum()) for d in displacement_set(R)}

    per_seed.append({
        "seed":seed,
        "local_selection":local_sel,
        "rule":{
            "radius":R,
            "K":K,
            "margin_max":mt,
            "energy_max":et,
            "support_min":st,
            "votes_min":vt,
            "validation_overrides":val_overrides,
        },
        "test":{
            "local_correct":local_correct,
            "attention_correct":final_correct,
            "delta":delta,
            "overrides":int(use.sum()),
            "rescues":rescue,
            "harms":harm,
            "top1_support_mean":float(Ste[:,0].mean()),
            "top1_abs_displacement_mean":float(np.abs(Dte[:,0]).mean()),
            "top1_displacement_hist":displacement_hist,
        },
    })

    deltas.append(delta)

    print(
        "SEED",seed,
        "R",R,
        "K",K,
        "LOCAL",local_correct,
        "ATTN",final_correct,
        "DELTA",delta,
        "RESCUE",rescue,
        "HARM",harm,
        "ABS_D",float(np.abs(Dte[:,0]).mean()),
        flush=True,
    )

local_sum=sum(r["test"]["local_correct"] for r in per_seed)
attn_sum=sum(r["test"]["attention_correct"] for r in per_seed)

report={
    "model":"v13-generator-displacement-attention",
    "benchmark_parent":"v12 SITE_EXACT 1719/1800",
    "geometry":{
        "active_order":"64 states in GEN=7 logical orbit order",
        "displacement":"logical index d; one logical step corresponds to one GEN=7 physical-row step",
        "radii":list(RADII),
        "alignment":"candidate x[(f+d) mod64] is aligned back to query feature f",
    },
    "value_relation":"exact Z256 equality only; Relation16 not used",
    "attention_readout":"top-K frozen local-model consensus",
    "rule_selection":"validation only",
    "test_tuning":False,
    "per_seed":per_seed,
    "aggregate":{
        "local_sum":local_sum,
        "attention_sum":attn_sum,
        "delta_sum":attn_sum-local_sum,
        "wins":sum(d>0 for d in deltas),
        "ties":sum(d==0 for d in deltas),
        "losses":sum(d<0 for d in deltas),
        "deltas":deltas,
    },
    "claim_boundary":(
        "v13 tests generator-ordered positional displacement on the 64 active digits probe. "
        "It does not yet establish the final U-Observer displacement law or map the 1,920 "
        "INFORMATION bytes."
    ),
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"generator_displacement_attention_v13.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
