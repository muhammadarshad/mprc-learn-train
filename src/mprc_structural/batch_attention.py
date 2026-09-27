"""Batch-exact implementation of the survived MPRC attention forward.

Input candidates: [N,C,128,113] uint8
Query:            [C,128,113] uint8

Returns one wide integer energy per candidate, summed over channels.
It is only an optimization of attention_forward and must stay bit-exact.
"""

from __future__ import annotations
import numpy as np
from .manifold import H,W,TILE_H,SLAB_COUNT,GEN
from .attention import validate_lut

ORDER64=np.asarray([(GEN*t)&63 for t in range(64)],dtype=np.int64)
ROW_ORDER=np.concatenate([ORDER64,64+ORDER64])


def transport_batch(x: np.ndarray) -> np.ndarray:
    a=np.asarray(x)
    if a.ndim!=4 or a.shape[2:]!=(H,W) or a.dtype!=np.uint8:
        raise ValueError("expected uint8 [N,C,128,113]")
    return a[:,:,ROW_ORDER,:].copy()


def react_batch(logical: np.ndarray,lut: np.ndarray|None=None,rounds:int=1)->np.ndarray:
    x=np.asarray(logical)
    if x.ndim!=4 or x.shape[2:]!=(H,W) or x.dtype!=np.uint8:
        raise ValueError("expected uint8 [N,C,128,113]")
    table=validate_lut(lut)
    r=int(rounds)
    if r<1: raise ValueError("rounds must be >=1")
    state=x.copy()
    for _ in range(r):
        out=state.copy()
        for s in range(SLAB_COUNT):
            lo=s*TILE_H; hi=lo+TILE_H
            slab=state[:,:,lo:hi,:]
            acc=(
                slab[:,:,1:-1,1:-1].astype(np.uint16)
                +slab[:,:,:-2,1:-1].astype(np.uint16)
                +slab[:,:,2:,1:-1].astype(np.uint16)
                +slab[:,:,1:-1,:-2].astype(np.uint16)
                +slab[:,:,1:-1,2:].astype(np.uint16)
            )&0xFF
            out[:,:,lo+1:hi-1,1:-1]=table[acc.astype(np.uint8)]
        state=out
    return state


def batch_energy(
    candidates:np.ndarray,
    query:np.ndarray,
    *,
    lut:np.ndarray|None=None,
    rounds:int=7,
)->np.ndarray:
    c=np.asarray(candidates)
    q=np.asarray(query)
    if c.ndim!=4 or c.shape[2:]!=(H,W) or c.dtype!=np.uint8:
        raise ValueError("candidates must be uint8 [N,C,128,113]")
    if q.ndim!=3 or q.shape[1:]!=(H,W) or q.dtype!=np.uint8:
        raise ValueError("query must be uint8 [C,128,113]")
    if c.shape[1]!=q.shape[0]:
        raise ValueError("channel mismatch")

    tc=transport_batch(c)
    tq=transport_batch(q[None,:,:,:])[0]
    bound=((tc.astype(np.uint16)+tq[None].astype(np.uint16))&0xFF).astype(np.uint8)
    reacted=react_batch(bound,lut=lut,rounds=rounds)

    aa=reacted.astype(np.int16)
    bb=tq[None].astype(np.int16)
    ab=(aa-bb)&0xFF
    ba=(bb-aa)&0xFF
    d=np.minimum(ab,ba)
    return d.sum(axis=(1,2,3),dtype=np.int64)
