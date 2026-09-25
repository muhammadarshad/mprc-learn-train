"""Pure Integer Training v1-A — Global Attention inside the trainable forward graph.

DECISION-GRADE A/B
==================
A0 local-only:
    byte -> trainable Z256 LUT -> integer linear readout

A1 GA-integrated:
    local byte
      -> Relation16 transpose policy
      -> sparse global occurrences in 14,464*N address space
      -> candidate sample collection
      -> ring MEASURE over complete 64-byte query/candidate state
      -> selected global context
      -> BIND query+context mod256
      -> generator-7 transported REACT input
      -> SAME trainable Z256 LUT
      -> integer linear readout

Both branches use:
    eta = 1/1000 represented by exact integer residual accumulators
    integer multiclass margin objective
    no softmax
    no float model state

INFORMATION SEMANTICS
=====================
The 1,920 INFORMATION bytes are NOT invented here. They remain neutral in this
compatibility probe. This script tests integration of GA + BIND + REACT + integer
learning, not the final information codec.

The optimized 64-feature REACT input is bit-exact gated against the reference
128x113 generator transport + identity REACT for the stated neutral embedding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

from mprc_structural.attention import (
    generator_orbit64,
    identity_lut,
    react_once_logical,
    transport_manifold,
)
from mprc_structural.global_attention import HV, Relation16TransposePolicy
from mprc_structural.manifold import H, W, DATA_W

Q=1000
SEED=20260925
EPOCHS=40
BATCH=64
MARGIN=16
NCLASS=10
FDIM=64
COL=1

assert DATA_W == 98
assert FDIM == 64

ORBIT=generator_orbit64()
SITE=np.asarray([int(ORBIT[t])*W + COL for t in range(FDIM)],dtype=np.int64)
assert len(set(map(int,SITE))) == FDIM


def trunc_div_array(a:np.ndarray,q:int)->np.ndarray:
    pos=a>=0
    out=np.empty_like(a,dtype=np.int64)
    out[pos]=a[pos]//q
    out[~pos]=-((-a[~pos])//q)
    return out


class ResidualParam:
    def __init__(self,value:np.ndarray,Q:int=Q,ring:bool=False):
        self.value=value.astype(np.int64,copy=True)
        self.residual=np.zeros_like(self.value,dtype=np.int64)
        self.Q=int(Q)
        self.ring=bool(ring)
        self.positive_quanta=0
        self.negative_quanta=0

    def step(self,grad:np.ndarray):
        self.residual += grad.astype(np.int64,copy=False)
        k=trunc_div_array(self.residual,self.Q)
        self.residual -= k*self.Q
        self.value -= k
        if self.ring:
            self.value &= 0xFF
        self.positive_quanta += int(np.maximum(k,0).sum())
        self.negative_quanta += int(np.maximum(-k,0).sum())
        assert np.all(np.abs(self.residual) < self.Q)


def tau16_bytes(x:np.ndarray)->np.ndarray:
    a=(x.astype(np.uint16)>>4)&0xF
    b=x.astype(np.uint16)&0xF
    return ((b<<4)|a).astype(np.uint8)


def cdist_rows(q:np.ndarray,X:np.ndarray)->np.ndarray:
    """sum_i cdist(q_i, X_ri), exact integer."""
    qa=q.astype(np.int16)[None,:]
    xa=X.astype(np.int16)
    ab=(qa-xa)&0xFF
    ba=(xa-qa)&0xFF
    return np.minimum(ab,ba).sum(axis=1,dtype=np.int64)


class ActiveGlobalTranspose:
    """Sparse occurrence index over the 64 active probe sites in full 14,464*N global space."""

    def __init__(self,Xmem:np.ndarray):
        self.X=np.asarray(Xmem,dtype=np.uint8)
        self.N=len(self.X)
        buckets=[[] for _ in range(256)]
        for n,row in enumerate(self.X):
            base=n*HV
            for f,p0 in enumerate(row):
                p=int(p0)
                g=base+int(SITE[f])
                buckets[p].append((g,n,f))
        self.buckets=buckets
        self.occurrences=self.N*FDIM

    def candidate_samples(self,q:np.ndarray,exclude_sample:int|None=None)->tuple[np.ndarray,int]:
        seen=set()
        occurrence_hits=0
        policy=Relation16TransposePolicy()
        for p0 in q:
            for p in policy.states(int(p0)):
                for g,n,f in self.buckets[p]:
                    occurrence_hits += 1
                    if exclude_sample is not None and n==exclude_sample:
                        continue
                    seen.add(n)
        if not seen:
            # Explicit fallback: all memory samples except self. This should be reported.
            seen=set(range(self.N))
            if exclude_sample is not None:
                seen.discard(exclude_sample)
        return np.asarray(sorted(seen),dtype=np.int64),occurrence_hits


def choose_contexts(Xquery:np.ndarray,index:ActiveGlobalTranspose,train_mode:bool):
    ctx=np.empty_like(Xquery)
    selected=np.empty(len(Xquery),dtype=np.int64)
    candidate_counts=[]
    occurrence_hits=[]
    energies=[]

    for qi,q in enumerate(Xquery):
        exclude=qi if train_mode else None
        cand,hits=index.candidate_samples(q,exclude_sample=exclude)
        E=cdist_rows(q,index.X[cand])
        # MEASURE argmin; deterministic sample-id tie break.
        order=np.lexsort((cand,E))
        best=int(cand[order[0]])
        ctx[qi]=index.X[best]
        selected[qi]=best
        candidate_counts.append(int(len(cand)))
        occurrence_hits.append(int(hits))
        energies.append(int(E[order[0]]))

    return ctx,selected,{
        "mean_candidate_samples":float(np.mean(candidate_counts)),
        "min_candidate_samples":int(np.min(candidate_counts)),
        "max_candidate_samples":int(np.max(candidate_counts)),
        "mean_occurrence_hits":float(np.mean(occurrence_hits)),
        "mean_selected_measure_energy":float(np.mean(energies)),
    }


def fast_react_indices(bound:np.ndarray)->np.ndarray:
    """Exact 64 active-site identity-REACT indices for neutral manifold embedding.

    Feature t is embedded at logical (row=t, col=1) after generator-7 transport.
    Other sites are neutral zero.

    v1 boundary rule:
      t=0,63 -> identity
      t=1..62 -> prev + center + next (left/right neutral zero)
    """
    x=np.asarray(bound,dtype=np.uint8)
    out=x.copy()
    if len(x):
        s=(
            x[:,0:-2].astype(np.uint16)
            + x[:,1:-1].astype(np.uint16)
            + x[:,2:].astype(np.uint16)
        )&0xFF
        out[:,1:-1]=s.astype(np.uint8)
    return out


def react_reference_gate():
    """Bit-exact comparison of optimized active-site indices to full reference topology."""
    rng=np.random.default_rng(991)
    trials=128
    for _ in range(trials):
        b=rng.integers(0,256,size=FDIM,dtype=np.uint8)

        physical=np.zeros((H,W),dtype=np.uint8)
        for t in range(FDIM):
            physical[int(ORBIT[t]),COL]=b[t]

        logical=transport_manifold(physical)
        reacted=react_once_logical(logical,lut=identity_lut())
        got=np.asarray([reacted[t,COL] for t in range(FDIM)],dtype=np.uint8)
        want=fast_react_indices(b[None,:])[0]
        assert np.array_equal(got,want)
    return trials


class IntegerNet:
    def __init__(self,rng:np.random.Generator,attention:bool):
        self.attention=bool(attention)

        # REACT/activation table is a true Z256 table.
        # Readout converts ring position relative to origin 128:
        #   <128 => positive, >128 => negative.
        self.L=ResidualParam(np.arange(256,dtype=np.int64),ring=True)

        self.W=ResidualParam(
            rng.integers(-2,3,size=(FDIM,NCLASS),dtype=np.int64)
        )
        self.b=ResidualParam(np.zeros(NCLASS,dtype=np.int64))

        self.lut_initial=self.L.value.copy()

    def input_indices(self,x:np.ndarray,ctx:np.ndarray|None)->np.ndarray:
        if not self.attention:
            return x.astype(np.uint8,copy=False)

        assert ctx is not None
        bound=((x.astype(np.uint16)+ctx.astype(np.uint16))&0xFF).astype(np.uint8)
        return fast_react_indices(bound)

    def features(self,idx:np.ndarray)->np.ndarray:
        ring=self.L.value[idx]
        # origin-128 polarity convention: below 128 positive, above 128 negative.
        return 128-ring

    def scores(self,x:np.ndarray,ctx:np.ndarray|None):
        idx=self.input_indices(x,ctx)
        a=self.features(idx)
        s=a@self.W.value + self.b.value
        return idx,a,s

    def train(self,Xtr,ytr,Xte,yte,ctx_tr=None,ctx_te=None):
        rng=np.random.default_rng(SEED+100+(1 if self.attention else 0))
        hist=[]

        for epoch in range(EPOCHS):
            ep=rng.permutation(len(Xtr))
            violations=0

            for st in range(0,len(ep),BATCH):
                ids=ep[st:st+BATCH]
                xb=Xtr[ids]
                yb=ytr[ids]
                cb=None if ctx_tr is None else ctx_tr[ids]

                idx,a,s=self.scores(xb,cb)
                ds=np.zeros_like(s,dtype=np.int64)

                for i in range(len(xb)):
                    yi=int(yb[i]); sy=int(s[i,yi]); v=0
                    for c in range(NCLASS):
                        if c==yi: continue
                        if int(s[i,c])+MARGIN>sy:
                            ds[i,c]+=1
                            v+=1
                    ds[i,yi]-=v
                    violations+=v

                gW=a.T@ds
                gb=ds.sum(axis=0,dtype=np.int64)
                ga=ds@self.W.value.T

                # a = 128 - L[idx], therefore dLoss/dL = -dLoss/da.
                gL=np.zeros(256,dtype=np.int64)
                np.add.at(gL,idx.reshape(-1),(-ga).reshape(-1))

                self.W.step(gW)
                self.b.step(gb)
                self.L.step(gL)

                self.W.value=np.clip(self.W.value,-127,127)
                self.b.value=np.clip(self.b.value,-32767,32767)
                # L is already wrapped into Z256 by ResidualParam.

            tr=self.evaluate(Xtr,ytr,ctx_tr)
            te=self.evaluate(Xte,yte,ctx_te)
            hist.append({
                "epoch":epoch+1,
                "violations":int(violations),
                "train_correct":tr[0],"train_total":tr[1],
                "test_correct":te[0],"test_total":te[1],
            })
            print(("GA" if self.attention else "LOCAL"),hist[-1],flush=True)

        return hist

    def evaluate(self,X,y,ctx=None):
        _,_,s=self.scores(X,ctx)
        p=s.argmax(axis=1)
        return int((p==y).sum()),len(y)

    def telemetry(self):
        return {
            "lut_changed_entries":int(np.count_nonzero(self.L.value!=self.lut_initial)),
            "lut_residual_abs_max":int(np.abs(self.L.residual).max()),
            "W_residual_abs_max":int(np.abs(self.W.residual).max()),
            "b_residual_abs_max":int(np.abs(self.b.residual).max()),
            "lut_positive_quanta":self.L.positive_quanta,
            "lut_negative_quanta":self.L.negative_quanta,
            "W_positive_quanta":self.W.positive_quanta,
            "W_negative_quanta":self.W.negative_quanta,
        }


# ------------------------------------------------------------------
# Math/code survival before labels enter training.
# ------------------------------------------------------------------
reference_trials=react_reference_gate()

# ------------------------------------------------------------------
# Real data
# ------------------------------------------------------------------
ds=load_digits()
X=np.asarray(ds.data,dtype=np.int64)
y=np.asarray(ds.target,dtype=np.int64)
Xbyte=(X*15).astype(np.uint8)

rng=np.random.default_rng(SEED)
order=rng.permutation(len(Xbyte))
cut=int(len(order)*0.7)
train_ix=order[:cut]
test_ix=order[cut:]

Xtr=Xbyte[train_ix]; ytr=y[train_ix]
Xte=Xbyte[test_ix]; yte=y[test_ix]

# Global memory contains training samples only.
index=ActiveGlobalTranspose(Xtr)
assert index.occurrences == len(Xtr)*FDIM

ctx_tr,sel_tr,diag_tr=choose_contexts(Xtr,index,train_mode=True)
ctx_te,sel_te,diag_te=choose_contexts(Xte,index,train_mode=False)

# Attention selection is label-free. Label agreement is reported only after selection.
train_neighbor_label_agree=int((ytr[sel_tr]==ytr).sum())
test_neighbor_label_agree=int((ytr[sel_te]==yte).sum())

# ------------------------------------------------------------------
# Same learner, two forward graphs.
# ------------------------------------------------------------------
local_net=IntegerNet(np.random.default_rng(SEED+1),attention=False)
ga_net=IntegerNet(np.random.default_rng(SEED+1),attention=True)

local_hist=local_net.train(Xtr,ytr,Xte,yte)
ga_hist=ga_net.train(Xtr,ytr,Xte,yte,ctx_tr,ctx_te)

local_final=local_net.evaluate(Xte,yte)
ga_final=ga_net.evaluate(Xte,yte,ctx_te)

report={
    "model":"pure-integer-v1A-global-attention-integration",
    "pretraining_gate":{
        "optimized_react_vs_full_reference_trials":reference_trials,
        "bit_exact":True,
    },
    "learning":{
        "learning_rate":{"numerator":1,"denominator":Q,"float_lr":False},
        "objective":"integer multiclass margin",
        "softmax":False,
        "model_state_float":False,
        "lut":"trainable Z256 256-entry REACT/activation table",
        "polarity_readout":"activation = 128 - LUT[state]",
    },
    "global_attention":{
        "local_domain":256,
        "global_domain":f"{HV}*N",
        "memory_samples":len(Xtr),
        "active_occurrences":index.occurrences,
        "global_address_rule":"g = sample*14464 + site",
        "site_assignment":"64 generator-7 row positions at DATA lane 1",
        "local_relation":"Relation16TransposePolicy p -> {p,tau(p)}",
        "candidate_rank":"sum cdist over all 64 query/candidate bytes",
        "selected_contexts":"one minimum-MEASURE training sample; label-free",
        "information_semantics":"NEUTRAL / NOT ACTIVE in v1-A",
        "train_retrieval":diag_tr,
        "test_retrieval":diag_te,
        "neighbor_label_agreement_diagnostic":{
            "train":[train_neighbor_label_agree,len(Xtr)],
            "test":[test_neighbor_label_agree,len(Xte)],
            "labels_used_for_selection":False,
        },
    },
    "ab":{
        "A0_local_only":{
            "test_correct":local_final[0],
            "test_total":local_final[1],
            "telemetry":local_net.telemetry(),
        },
        "A1_global_attention":{
            "test_correct":ga_final[0],
            "test_total":ga_final[1],
            "telemetry":ga_net.telemetry(),
        },
    },
    "local_history":local_hist,
    "ga_history":ga_hist,
    "claim_boundary":(
        "v1-A integrates sparse local/global retrieval, ring MEASURE context selection, "
        "BIND, generator-7 REACT inputs, a trainable Z256 LUT, integer residual learning "
        "and no softmax. The 1,920 INFORMATION-byte semantics are deliberately neutral, "
        "and attention selection itself is not yet learned."
    )
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"pure_integer_training_v1a_ga.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
