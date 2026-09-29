"""Synchronous parallel directional projection for MPRC states.

Candidate execution primitive, not yet a frozen semantic rule.

One resolved snapshot X_t is observed by all directional branches:
    LR, RL, C->L, C->R

Each branch may internally resolve along its own traversal, but no branch may
observe another branch's provisional mutations.

Only after all four projections complete are proposals reconciled:
    X_{t+1} = VOTE(P_LR, P_RL, P_CL, P_CR ; X_t)

This module freezes the SYNCHRONY contract only. The concrete directional
resolver used by a model remains an empirical/candidate choice.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

DIRECTIONS=("LR","RL","CL","CR")

def cdist(a:int,b:int)->int:
    a=int(a)&255;b=int(b)&255
    return min((a-b)&255,(b-a)&255)

def ring_medoid(values:list[int], incumbent:int)->int:
    """Vote among proposed Z256 states by minimum total circular distance.

    Ties preserve incumbent when it is tied; otherwise smallest byte for
    deterministic testing. This is a candidate vote rule, not frozen MPRC law.
    """
    vals=[int(v)&255 for v in values]
    cand=sorted(set(vals+[int(incumbent)&255]))
    scored=[(sum(cdist(c,v) for v in vals),c) for c in cand]
    best_e=min(e for e,_ in scored)
    tied=[c for e,c in scored if e==best_e]
    inc=int(incumbent)&255
    return inc if inc in tied else min(tied)

def directional_orders(width:int)->dict[str,list[int]]:
    W=int(width)
    if W<1: raise ValueError("width must be positive")
    lr=list(range(W))
    rl=list(range(W-1,-1,-1))
    # center-out is split into two independent projections.
    # For even W the central pair are W/2-1 and W/2.
    left=(W-1)//2
    right=W//2
    cl=list(range(left,-1,-1))
    cr=list(range(right,W))
    return {"LR":lr,"RL":rl,"CL":cl,"CR":cr}

@dataclass(frozen=True)
class ProjectionResult:
    direction:str
    proposal:np.ndarray
    mutated:np.ndarray

Resolver=Callable[[int,int],int]

def pair_resolver(cur:int,context:int)->int:
    """Conservative candidate resolver for gates/diagnostics.

    Choose the ring-medoid of current and context. With two points there is
    generally a tie, so incumbent is retained. Model experiments should supply
    an explicit resolver; this default intentionally avoids invented mutation.
    """
    return ring_medoid([cur,context],cur)

def project_direction(snapshot:np.ndarray,direction:str,resolver:Resolver=pair_resolver)->ProjectionResult:
    """Resolve one directional branch from an immutable 1-D snapshot.

    The branch gets a private working copy. Internal traversal can therefore
    use states resolved earlier IN THAT BRANCH, while other branches remain
    isolated and continue to observe the original snapshot.
    """
    x=np.asarray(snapshot)
    if x.ndim!=1 or x.dtype!=np.uint8:
        raise ValueError("snapshot must be uint8 [W]")
    if direction not in DIRECTIONS: raise ValueError(direction)
    work=x.copy()
    order=directional_orders(len(x))[direction]

    # LR/RL use predecessor in traversal. Center branches begin at center and
    # propagate outward. First site has no context and remains unchanged.
    for k in range(1,len(order)):
        i=order[k]; prev=order[k-1]
        work[i]=int(resolver(int(work[i]),int(work[prev])))&255
    return ProjectionResult(direction,work,work!=x)

def vote(
    snapshot:np.ndarray,
    proposals:dict[str,np.ndarray],
    branch_mutations:dict[str,np.ndarray],
)->tuple[np.ndarray,np.ndarray]:
    """Resolve only among branches that actually proposed mutation.

    KEEP is not a vote against mutation. If no branch proposes a mutation,
    preserve the incumbent. If one or more branches propose, reconcile only
    those proposed states. This matches the synchronous rule:
        observe in parallel -> collect mutations -> vote -> update once.
    """
    x=np.asarray(snapshot)
    if set(proposals)!=set(DIRECTIONS) or set(branch_mutations)!=set(DIRECTIONS):
        raise ValueError("need all four directions")
    out=x.copy();changed=np.zeros(len(x),dtype=np.bool_)
    for i in range(len(x)):
        vals=[
            int(proposals[d][i])
            for d in DIRECTIONS
            if bool(branch_mutations[d][i])
        ]
        if not vals:
            continue

        # Exact plurality first.
        uniq=sorted(set(vals))
        counts={v:vals.count(v) for v in uniq}
        max_count=max(counts.values())
        tied=[v for v in uniq if counts[v]==max_count]

        if len(tied)==1:
            z=tied[0]
        else:
            # Tie: choose the proposal minimizing circular distance to the
            # other mutation proposals. Incumbent is NOT a candidate.
            scored=[
                (sum(cdist(c,v) for v in vals), cdist(c,int(x[i])), c)
                for c in tied
            ]
            z=min(scored)[2]

        out[i]=z
        changed[i]=(z!=int(x[i]))
    return out,changed

def parallel_resolve(snapshot:np.ndarray,resolver:Resolver=pair_resolver,execution_order=None)->dict:
    """Run four projections from the SAME X_t, then one vote.

    execution_order exists only to prove schedule independence.
    """
    x=np.asarray(snapshot)
    if x.ndim!=1 or x.dtype!=np.uint8:
        raise ValueError("snapshot must be uint8 [W]")
    schedule=list(DIRECTIONS if execution_order is None else execution_order)
    if sorted(schedule)!=sorted(DIRECTIONS): raise ValueError("bad execution order")

    proposals={}
    branch_mutations={}
    # Crucial: every call receives x, never a previously projected state.
    for d in schedule:
        r=project_direction(x,d,resolver)
        proposals[d]=r.proposal
        branch_mutations[d]=r.mutated

    nxt,mut=vote(x,proposals,branch_mutations)
    return {
        "snapshot":x.copy(),
        "proposals":{d:proposals[d] for d in DIRECTIONS},
        "branch_mutations":{d:branch_mutations[d] for d in DIRECTIONS},
        "state":nxt,
        "mutated":mut,
    }
