"""v1-B Attention Integration Loss Localization.

The v1-A probe showed:
- GA-selected nearest global context has strong label agreement.
- collapsing query+context into one Z256 byte stream then REACTing it destroys much of that signal.

This probe isolates the integration operator while keeping:
- the SAME train/test split,
- the SAME label-free GA-selected contexts,
- the SAME integer residual learner,
- the SAME integer multiclass margin objective,
- no softmax,
- no float model state.

Branches
--------
A0  Q          : query/local only
A1  C          : selected global context only
A2  W2=(Q,C)   : exact pair-preserving relation, 128 byte features
A3  T=(S,D)    : Arshad transpose coordinates
                 S=Q+C mod256, D=Q-C mod256, 128 byte features
A4  B=Q+C      : scalar bind only, 64 byte features
A5  R(B)       : current v1-A scalar bind + one REACT round, 64 byte features

The key falsification:
If A2 succeeds while A4/A5 fail, scalar collapse is the problem.
If A3 loses against A2, the known 2-to-1 transpose kernel matters.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits

from mprc_structural.global_attention import HV, Relation16TransposePolicy
from mprc_structural.manifold import W as MAN_W, DATA_W, GEN
from mprc_structural.attention import generator_orbit64

QDEN=1000
SEED=20260925
EPOCHS=40
BATCH=64
MARGIN=16
NCLASS=10
FDIM=64
COL=1

ORBIT=generator_orbit64()
SITE=np.asarray([int(ORBIT[t])*MAN_W + COL for t in range(FDIM)],dtype=np.int64)


def trunc_div_array(a:np.ndarray,q:int)->np.ndarray:
    pos=a>=0
    out=np.empty_like(a,dtype=np.int64)
    out[pos]=a[pos]//q
    out[~pos]=-((-a[~pos])//q)
    return out


class ResidualParam:
    def __init__(self,value:np.ndarray,Q:int=QDEN,ring:bool=False):
        self.value=value.astype(np.int64,copy=True)
        self.residual=np.zeros_like(self.value,dtype=np.int64)
        self.Q=int(Q)
        self.ring=bool(ring)

    def step(self,grad:np.ndarray):
        self.residual += grad.astype(np.int64,copy=False)
        k=trunc_div_array(self.residual,self.Q)
        self.residual -= k*self.Q
        self.value -= k
        if self.ring:
            self.value &= 255
        assert np.all(np.abs(self.residual)<self.Q)


def cdist_rows(q:np.ndarray,X:np.ndarray)->np.ndarray:
    qa=q.astype(np.int16)[None,:]
    xa=X.astype(np.int16)
    ab=(qa-xa)&255
    ba=(xa-qa)&255
    return np.minimum(ab,ba).sum(axis=1,dtype=np.int64)


class ActiveGlobalTranspose:
    def __init__(self,Xmem:np.ndarray):
        self.X=np.asarray(Xmem,dtype=np.uint8)
        self.N=len(self.X)
        buckets=[[] for _ in range(256)]
        for n,row in enumerate(self.X):
            base=n*HV
            for f,p0 in enumerate(row):
                p=int(p0)
                buckets[p].append((base+int(SITE[f]),n,f))
        self.buckets=buckets

    def candidate_samples(self,q:np.ndarray,exclude_sample=None):
        seen=set()
        policy=Relation16TransposePolicy()
        for p0 in q:
            for p in policy.states(int(p0)):
                for g,n,f in self.buckets[p]:
                    if exclude_sample is not None and n==exclude_sample:
                        continue
                    seen.add(n)
        if not seen:
            seen=set(range(self.N))
            if exclude_sample is not None:
                seen.discard(exclude_sample)
        return np.asarray(sorted(seen),dtype=np.int64)


def choose_contexts(Xquery,index,train_mode):
    ctx=np.empty_like(Xquery)
    selected=np.empty(len(Xquery),dtype=np.int64)
    for qi,q in enumerate(Xquery):
        cand=index.candidate_samples(q,exclude_sample=(qi if train_mode else None))
        E=cdist_rows(q,index.X[cand])
        order=np.lexsort((cand,E))
        best=int(cand[order[0]])
        ctx[qi]=index.X[best]
        selected[qi]=best
    return ctx,selected


def react64(bound:np.ndarray)->np.ndarray:
    x=np.asarray(bound,dtype=np.uint8)
    out=x.copy()
    s=(x[:,:-2].astype(np.uint16)+x[:,1:-1].astype(np.uint16)+x[:,2:].astype(np.uint16))&255
    out[:,1:-1]=s.astype(np.uint8)
    return out


def branch_indices(name,q,c):
    if name=="Q":
        return q
    if name=="C":
        return c
    if name=="W2":
        return np.concatenate([q,c],axis=1)
    if name=="SD":
        S=((q.astype(np.uint16)+c.astype(np.uint16))&255).astype(np.uint8)
        D=((q.astype(np.int16)-c.astype(np.int16))&255).astype(np.uint8)
        return np.concatenate([S,D],axis=1)
    if name=="B":
        return ((q.astype(np.uint16)+c.astype(np.uint16))&255).astype(np.uint8)
    if name=="RB":
        B=((q.astype(np.uint16)+c.astype(np.uint16))&255).astype(np.uint8)
        return react64(B)
    raise KeyError(name)


class Net:
    def __init__(self,feature_dim,rng):
        self.L=ResidualParam(np.arange(256,dtype=np.int64),ring=True)
        self.W=ResidualParam(rng.integers(-2,3,size=(feature_dim,NCLASS),dtype=np.int64))
        self.b=ResidualParam(np.zeros(NCLASS,dtype=np.int64))
        self.lut0=self.L.value.copy()

    def scores_from_idx(self,idx):
        ring=self.L.value[idx]
        a=128-ring
        return idx,a,a@self.W.value+self.b.value

    def eval(self,idx,y):
        _,_,s=self.scores_from_idx(idx)
        p=s.argmax(axis=1)
        return int((p==y).sum()),len(y)

    def train(self,itr,ytr,ite,yte):
        rng=np.random.default_rng(SEED+123+itr.shape[1])
        best=(0,0)
        history=[]
        for epoch in range(EPOCHS):
            ep=rng.permutation(len(itr))
            vio=0
            for st in range(0,len(ep),BATCH):
                ids=ep[st:st+BATCH]
                idx=itr[ids]; yb=ytr[ids]
                _,a,s=self.scores_from_idx(idx)
                ds=np.zeros_like(s,dtype=np.int64)
                for i in range(len(idx)):
                    yi=int(yb[i]); sy=int(s[i,yi]); v=0
                    for cc in range(NCLASS):
                        if cc==yi: continue
                        if int(s[i,cc])+MARGIN>sy:
                            ds[i,cc]+=1; v+=1
                    ds[i,yi]-=v; vio+=v
                gW=a.T@ds
                gb=ds.sum(axis=0,dtype=np.int64)
                ga=ds@self.W.value.T
                gL=np.zeros(256,dtype=np.int64)
                np.add.at(gL,idx.reshape(-1),(-ga).reshape(-1))
                self.W.step(gW); self.b.step(gb); self.L.step(gL)
                self.W.value=np.clip(self.W.value,-127,127)
                self.b.value=np.clip(self.b.value,-32767,32767)

            te=self.eval(ite,yte)
            tr=self.eval(itr,ytr)
            if te[0]>best[0]: best=(te[0],epoch+1)
            history.append({
                "epoch":epoch+1,
                "train_correct":tr[0],
                "train_total":tr[1],
                "test_correct":te[0],
                "test_total":te[1],
                "violations":int(vio),
            })
        final=self.eval(ite,yte)
        return {
            "final_test":[final[0],final[1]],
            "best_test":[best[0],len(yte)],
            "best_epoch":best[1],
            "lut_changed_entries":int(np.count_nonzero(self.L.value!=self.lut0)),
            "feature_dim":int(itr.shape[1]),
            "history":history,
        }


# Data and frozen split.
ds=load_digits()
X=np.asarray(ds.data,dtype=np.int64)
y=np.asarray(ds.target,dtype=np.int64)
Xb=(X*15).astype(np.uint8)
rng=np.random.default_rng(SEED)
order=rng.permutation(len(Xb))
cut=int(len(order)*0.7)
trix=order[:cut]; teix=order[cut:]
Xtr=Xb[trix];ytr=y[trix]
Xte=Xb[teix];yte=y[teix]

# One label-free GA context selection reused by every branch.
index=ActiveGlobalTranspose(Xtr)
Ctr,seltr=choose_contexts(Xtr,index,True)
Cte,selte=choose_contexts(Xte,index,False)
neighbor_test=int((ytr[selte]==yte).sum())

branches={}
for bi,name in enumerate(("Q","C","W2","SD","B","RB")):
    Itr=branch_indices(name,Xtr,Ctr)
    Ite=branch_indices(name,Xte,Cte)
    net=Net(Itr.shape[1],np.random.default_rng(SEED+bi))
    res=net.train(Itr,ytr,Ite,yte)
    branches[name]=res
    print(name,res["final_test"],"best",res["best_test"],"epoch",res["best_epoch"],flush=True)

# Exact representation diagnostics.
# W2 always reconstructs q,c trivially.
# S,D has two-preimage ambiguity under simultaneous half-turn.
rng2=np.random.default_rng(77)
pairs=rng2.integers(0,256,size=(100000,2),dtype=np.uint16)
S=(pairs[:,0]+pairs[:,1])&255
D=(pairs[:,0]-pairs[:,1])&255
pairs2=(pairs+128)&255
S2=(pairs2[:,0]+pairs2[:,1])&255
D2=(pairs2[:,0]-pairs2[:,1])&255
assert np.array_equal(S,S2)
assert np.array_equal(D,D2)

report={
    "model":"v1B-attention-integration-loss-localization",
    "ga_context":{
        "policy":"Relation16TransposePolicy",
        "measure":"64-byte sum cdist",
        "test_neighbor_label_agreement":[neighbor_test,len(yte)],
        "labels_used_for_selection":False,
    },
    "branches":{
        "Q":"query/local only",
        "C":"selected global context only",
        "W2":"exact pair-preserving (query,context)",
        "SD":"(sum mod256, difference mod256); two-preimage half-turn ambiguity",
        "B":"scalar bind query+context mod256",
        "RB":"scalar bind + one REACT round",
    },
    "results":branches,
    "representation_gate":{
        "W2_exact_pair":True,
        "SD_simultaneous_half_turn_collision":True,
        "SD_random_collision_witnesses":len(pairs),
    },
    "learning":{
        "integer_residual_denominator":QDEN,
        "softmax":False,
        "float_model_state":False,
        "trainable_Z256_LUT":True,
    },
    "claim_boundary":(
        "This is a controlled integration ablation using one fixed label-free GA context selection. "
        "It localizes information loss among pair preservation, transpose coordinates, scalar bind, "
        "and REACT. It does not define final INFORMATION semantics or final MPRC learning law."
    )
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"attention_integration_loss_localization_v1b.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
