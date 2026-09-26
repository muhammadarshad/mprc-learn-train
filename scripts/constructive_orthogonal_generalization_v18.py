"""v18 — Constructive orthogonal-feature generalization.

Research question
=================
Can the model construct a response for a combination it has NEVER stored, rather
than behaving as holographic memory over whole feature combinations?

Real source data:
    sklearn handwritten digits.

Controlled composition:
    two disjoint 64-state factor coordinates:
        A = first factor state
        B = second factor state

The ordered response is:
        R = (class(A), class(B))

Combination split
=================
All 100 ordered digit combinations exist conceptually.

SEEN pair combinations:
    (a+b) mod 2 == 0       -> 50 combinations

UNSEEN pair combinations:
    (a+b) mod 2 == 1       -> 50 combinations

Every digit 0..9 occurs in both roles and in both SEEN and UNSEEN sets.
Thus individual factors are known, but half the combinations are never placed in
pair memory.

Constructive MPRC-style path
============================
For each factor independently:
    local byte state
      -> frozen integer local model
      -> position-aware exact-state attention over TRAIN digit memory
      -> validation-selected second opinion
      -> SELECT factor identity

Then construct:
    response = (SELECT_A, SELECT_B)

There are NO pair-level learned parameters and NO pair label memory in this path.

Holographic/full-state memory control
=====================================
Store only SEEN composite states:
    H = [A || B]  (128 bytes)
with their pair labels.

Retrieve nearest full composite by ring cdist.
Since UNSEEN pair labels are absent from memory, this control cannot construct a
new pair label; it can only return a remembered combination.

This contrast is intentional. It tests construction vs whole-pattern recall.

Claim boundary
==============
The digit pairing itself is a controlled compositional benchmark, not a natural
scene. Success demonstrates held-out combination construction, not general AGI or
a final MPRC INFORMATION codec.
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
KS=(1,3,5,7)
PAIR_REPLICATES_TRAIN=5
PAIR_REPLICATES_VAL=5
PAIR_REPLICATES_TEST=10


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

                self.W.step(gW); self.b.step(gb); self.L.step(gL)
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


def score_stats(scores):
    order=np.argsort(scores,axis=1)
    p1=order[:,-1]; p2=order[:,-2]
    r=np.arange(len(scores))
    margin=scores[r,p1]-scores[r,p2]
    return p1.astype(np.int64),margin.astype(np.int64)


def quantiles(x,qs):
    z=np.sort(np.asarray(x,dtype=np.int64))
    out=[]
    for q in qs:
        idx=min(len(z)-1,(len(z)*q)//100)
        out.append(int(z[idx]))
    return sorted(set(out))


class PositionExactAttention:
    """Same-site exact-state support + ring MEASURE ranking."""

    def __init__(self,Xmem):
        self.X=np.asarray(Xmem,dtype=np.uint8)

    def topk(self,Q,maxk=7,exclude_self=False):
        n=len(Q)
        ids=np.empty((n,maxk),dtype=np.int64)
        E=np.empty((n,maxk),dtype=np.int64)
        S=np.empty((n,maxk),dtype=np.int64)

        for i,q in enumerate(Q):
            support=(self.X==q[None,:]).sum(axis=1,dtype=np.int64)
            energy=cdist_rows(q,self.X)
            cand=np.arange(len(self.X),dtype=np.int64)

            if exclude_self and i<len(self.X):
                keep=cand!=i
                cand=cand[keep]
                support=support[keep]
                energy=energy[keep]

            order=np.lexsort((cand,energy,-support))
            take=min(maxk,len(order))
            pick=order[:take]

            ids[i,:take]=cand[pick]
            E[i,:take]=energy[pick]
            S[i,:take]=support[pick]

            if take<maxk:
                ids[i,take:]=ids[i,take-1]
                E[i,take:]=E[i,take-1]
                S[i,take:]=S[i,take-1]

        return ids,E,S


def context_predictions(net,attn,ids):
    flat=attn.X[ids.reshape(-1)]
    return net.predict(flat).reshape(ids.shape)


def consensus(pred,E,K):
    P=pred[:,:K]
    Z=E[:,:K]
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

        out[i]=w
        votes[i]=vmax

    return out,votes


def apply_factor_rule(pq,pa,margin,e1,s1,votes,K,mt,et,st,vt):
    use=(
        (pq!=pa)
        & (margin<=mt)
        & (e1<=et)
        & (s1>=st)
        & (votes>=vt)
    )
    out=pq.copy()
    out[use]=pa[use]
    return out,use


def choose_factor_rule(net,attn,Xtr,Xva,yva):
    _,str_=net.scores(Xtr)
    _,mtr=score_stats(str_)

    _,sva=net.scores(Xva)
    pqva,mva=score_stats(sva)

    idtr,Etr,Str=attn.topk(Xtr,maxk=max(KS),exclude_self=True)
    idva,Eva,Sva=attn.topk(Xva,maxk=max(KS),exclude_self=False)

    predva=context_predictions(net,attn,idva)

    MTS=quantiles(mtr,(10,25,50,75,90,100))
    ETS=quantiles(Etr[:,0],(25,50,75,90,100))
    STS=quantiles(Str[:,0],(0,25,50,75,90,100))

    best=None

    for K in KS:
        pa,votes=consensus(predva,Eva,K)
        for mt in MTS:
            for et in ETS:
                for st in STS:
                    for vt in range(1,K+1):
                        p,use=apply_factor_rule(
                            pqva,pa,mva,Eva[:,0],Sva[:,0],votes,
                            K,mt,et,st,vt
                        )
                        ac=int((p==yva).sum())
                        key=(ac,-int(use.sum()),int(st),int(vt),-int(K))

                        if best is None or key>best[0]:
                            best=(key,(K,mt,et,st,vt))

    return best[1]


def factor_predict(net,attn,X,rule):
    K,mt,et,st,vt=rule

    _,s=net.scores(X)
    pq,m=score_stats(s)

    ids,E,S=attn.topk(X,maxk=max(KS),exclude_self=False)
    pred=context_predictions(net,attn,ids)
    pa,votes=consensus(pred,E,K)

    out,use=apply_factor_rule(
        pq,pa,m,E[:,0],S[:,0],votes,
        K,mt,et,st,vt
    )
    return out,use


def class_buckets(X,y,seed):
    rng=np.random.default_rng(seed)
    out={}
    for c in range(NCLASS):
        z=X[y==c]
        out[c]=z[rng.permutation(len(z))]
    return out


SEEN=[(a,b) for a in range(NCLASS) for b in range(NCLASS) if (a+b)%2==0]
UNSEEN=[(a,b) for a in range(NCLASS) for b in range(NCLASS) if (a+b)%2==1]

assert len(SEEN)==50
assert len(UNSEEN)==50

for d in range(NCLASS):
    assert any(a==d for a,b in SEEN)
    assert any(b==d for a,b in SEEN)
    assert any(a==d for a,b in UNSEEN)
    assert any(b==d for a,b in UNSEEN)


def make_pairs(buckets,combos,reps):
    A=[];B=[];Y=[]

    for a,b in combos:
        ZA=buckets[a]
        ZB=buckets[b]

        for r in range(reps):
            ia=r%len(ZA)
            ib=(3*r + a + b)%len(ZB)
            A.append(ZA[ia])
            B.append(ZB[ib])
            Y.append((a,b))

    return (
        np.asarray(A,dtype=np.uint8),
        np.asarray(B,dtype=np.uint8),
        np.asarray(Y,dtype=np.int64),
    )


def ring_pair_memory_predict(memA,memB,memY,A,B):
    pred=np.empty((len(A),2),dtype=np.int64)

    for i,(a,b) in enumerate(zip(A,B)):
        eA=cdist_rows(a,memA)
        eB=cdist_rows(b,memB)
        E=eA+eB
        j=int(np.argmin(E))
        pred[i]=memY[j]

    return pred


# Fixed factor split.
ds=load_digits()
X=(np.asarray(ds.data,dtype=np.int64)*15).astype(np.uint8)
y=np.asarray(ds.target,dtype=np.int64)

outer=StratifiedShuffleSplit(n_splits=1,test_size=.20,random_state=SPLIT_SEED)
tv,te=next(outer.split(X,y))

inner=StratifiedShuffleSplit(n_splits=1,test_size=.25,random_state=SPLIT_SEED+1)
trr,var=next(inner.split(X[tv],y[tv]))
tr=tv[trr]
va=tv[var]

Xtr=X[tr];ytr=y[tr]
Xva=X[va];yva=y[va]
Xte=X[te];yte=y[te]

train_buckets=class_buckets(Xtr,ytr,SPLIT_SEED+10)
val_buckets=class_buckets(Xva,yva,SPLIT_SEED+20)
test_buckets=class_buckets(Xte,yte,SPLIT_SEED+30)

# Pair memory contains SEEN combinations only.
memA,memB,memY=make_pairs(train_buckets,SEEN,PAIR_REPLICATES_TRAIN)
seenVA,seenVB,seenVY=make_pairs(val_buckets,SEEN,PAIR_REPLICATES_VAL)
unseenA,unseenB,unseenY=make_pairs(test_buckets,UNSEEN,PAIR_REPLICATES_TEST)

# Structural gate: no held-out pair label is present in pair memory.
memory_label_set=set(map(tuple,memY.tolist()))
assert all(tuple(z) in set(SEEN) for z in memory_label_set)
assert all(tuple(z) not in memory_label_set for z in unseenY)

per_seed=[]

for seed in MODEL_SEEDS:
    rng=np.random.default_rng(seed)
    W0=rng.integers(-2,3,size=(FDIM,NCLASS),dtype=np.int64)
    b0=np.zeros(NCLASS,dtype=np.int64)
    L0=np.arange(256,dtype=np.int64)

    rr=np.random.default_rng(seed+100000)
    orders=[rr.permutation(len(Xtr)) for _ in range(EPOCHS)]

    net=LocalNet(W0,b0,L0)
    local_sel=net.train_select_validation(Xtr,ytr,Xva,yva,orders)

    attn=PositionExactAttention(Xtr)
    factor_rule=choose_factor_rule(net,attn,Xtr,Xva,yva)

    # Single-factor held-out test.
    factor_test_pred,_=factor_predict(net,attn,Xte,factor_rule)
    factor_test_correct=int((factor_test_pred==yte).sum())

    # Constructive unseen pair response.
    pA,useA=factor_predict(net,attn,unseenA,factor_rule)
    pB,useB=factor_predict(net,attn,unseenB,factor_rule)
    constructed=np.stack([pA,pB],axis=1)

    pair_exact=int(np.all(constructed==unseenY,axis=1).sum())
    A_correct=int((pA==unseenY[:,0]).sum())
    B_correct=int((pB==unseenY[:,1]).sum())

    # Whole-pattern memory baseline.
    mem_seen_pred=ring_pair_memory_predict(memA,memB,memY,seenVA,seenVB)
    mem_seen_exact=int(np.all(mem_seen_pred==seenVY,axis=1).sum())

    mem_unseen_pred=ring_pair_memory_predict(memA,memB,memY,unseenA,unseenB)
    mem_unseen_exact=int(np.all(mem_unseen_pred==unseenY,axis=1).sum())
    mem_unseen_A=int((mem_unseen_pred[:,0]==unseenY[:,0]).sum())
    mem_unseen_B=int((mem_unseen_pred[:,1]==unseenY[:,1]).sum())

    # Every held-out combination must be produced at least once correctly if the
    # constructor truly spans the orthogonal product, not only a subset.
    combo_success={}
    for pair in UNSEEN:
        m=np.all(unseenY==np.asarray(pair)[None,:],axis=1)
        combo_success[f"{pair[0]},{pair[1]}"]=int(np.all(constructed[m]==unseenY[m],axis=1).sum())

    combos_with_success=sum(v>0 for v in combo_success.values())

    per_seed.append({
        "seed":seed,
        "local_selection":local_sel,
        "factor_attention_rule":{
            "K":factor_rule[0],
            "margin_max":factor_rule[1],
            "energy_max":factor_rule[2],
            "support_min":factor_rule[3],
            "votes_min":factor_rule[4],
        },
        "single_factor_test":{
            "correct":factor_test_correct,
            "total":len(yte),
        },
        "constructive_unseen_pairs":{
            "exact_pair_correct":pair_exact,
            "pair_total":len(unseenY),
            "factor_A_correct":A_correct,
            "factor_B_correct":B_correct,
            "factor_total":len(unseenY),
            "attention_overrides_A":int(useA.sum()),
            "attention_overrides_B":int(useB.sum()),
            "heldout_combinations_with_at_least_one_exact_success":combos_with_success,
            "heldout_combination_count":len(UNSEEN),
        },
        "whole_pattern_memory_control":{
            "seen_pair_validation_exact":mem_seen_exact,
            "seen_pair_validation_total":len(seenVY),
            "unseen_pair_exact":mem_unseen_exact,
            "unseen_pair_total":len(unseenY),
            "unseen_factor_A_correct":mem_unseen_A,
            "unseen_factor_B_correct":mem_unseen_B,
        },
    })

    print(
        "SEED",seed,
        "FACTOR",factor_test_correct,"/",len(yte),
        "CONSTRUCT_UNSEEN",pair_exact,"/",len(unseenY),
        "COMBOS",combos_with_success,"/",len(UNSEEN),
        "MEM_SEEN",mem_seen_exact,"/",len(seenVY),
        "MEM_UNSEEN",mem_unseen_exact,"/",len(unseenY),
        flush=True,
    )


construct_sum=sum(r["constructive_unseen_pairs"]["exact_pair_correct"] for r in per_seed)
factorA_sum=sum(r["constructive_unseen_pairs"]["factor_A_correct"] for r in per_seed)
factorB_sum=sum(r["constructive_unseen_pairs"]["factor_B_correct"] for r in per_seed)
memory_unseen_sum=sum(r["whole_pattern_memory_control"]["unseen_pair_exact"] for r in per_seed)
memory_seen_sum=sum(r["whole_pattern_memory_control"]["seen_pair_validation_exact"] for r in per_seed)

report={
    "model":"v18-constructive-orthogonal-feature-generalization",
    "principle":"attend/select factors independently, then construct response; do not memorize whole combinations",
    "factor_coordinates":{
        "A":"64-state digit factor",
        "B":"independent 64-state digit factor",
        "composite":"ordered pair (A,B); no pair-level parameters in constructive path",
    },
    "combination_split":{
        "seen_rule":"(a+b) mod2 == 0",
        "unseen_rule":"(a+b) mod2 == 1",
        "seen_combinations":len(SEEN),
        "unseen_combinations":len(UNSEEN),
        "all_factor_classes_present_in_both_roles":True,
    },
    "constructive_path":"factor local model + position-exact attention + SELECT_A/SELECT_B + tuple construction",
    "whole_pattern_memory_control":"ring-cdist nearest stored 128-byte composite; SEEN pairs only",
    "pair_memory_contains_unseen_labels":False,
    "per_seed":per_seed,
    "aggregate":{
        "constructive_unseen_pair_correct_sum":construct_sum,
        "constructive_unseen_pair_total":len(MODEL_SEEDS)*len(unseenY),
        "constructive_factor_A_correct_sum":factorA_sum,
        "constructive_factor_B_correct_sum":factorB_sum,
        "whole_pattern_memory_seen_correct_sum":memory_seen_sum,
        "whole_pattern_memory_seen_total":len(MODEL_SEEDS)*len(seenVY),
        "whole_pattern_memory_unseen_correct_sum":memory_unseen_sum,
        "whole_pattern_memory_unseen_total":len(MODEL_SEEDS)*len(unseenY),
    },
    "claim_boundary":(
        "v18 is a controlled combinatorial-generalization benchmark built from real "
        "handwritten digit states. The pair composition is constructed by the experiment. "
        "Success demonstrates response construction over held-out factor combinations; "
        "it does not yet establish natural-scene compositional reasoning or the final "
        "MPRC INFORMATION/U-observer architecture."
    ),
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"constructive_orthogonal_generalization_v18.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
