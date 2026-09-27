"""Coherent-displacement SELECT for directional ADI descriptor fields.

A byte/local descriptor says what occupies a local region.  Global identity
requires that many local occurrences agree under ONE common displacement.

For query Q[ch,r,c] and memory M_n[ch,r,c], define

    S_n(dr,dc) =
        # {(ch,r,c): Q[ch,r,c] == M_n[ch,r+dr,c+dc]}

over the overlapping anchor lattice, with exact Z256^9 descriptor equality.

Then

    M_n(Q) = max_(dr,dc) S_n(dr,dc)

and retain the displacement(s) attaining the maximum.  This is translation
compatible, but NOT invariant to arbitrary permutation of occurrence positions.
No labels, thresholds, Top-K, softmax, or learned weights.
"""
from __future__ import annotations
import numpy as np


def coherent_displacement_scores(
    query: np.ndarray, memory: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (best_support[N], best_displacement[N,2]).

    query:  uint8 [C,H,W,9]
    memory: uint8 [N,C,H,W,9]
    """
    q=np.asarray(query)
    m=np.asarray(memory)
    if q.ndim!=4 or q.shape[-1]!=9 or q.dtype!=np.uint8:
        raise ValueError("query must be uint8 [C,H,W,9]")
    if m.ndim!=5 or m.shape[1:]!=q.shape or m.dtype!=np.uint8:
        raise ValueError("memory must be uint8 [N,C,H,W,9] matching query")
    N,C,H,W,_=m.shape
    best=np.zeros(N,dtype=np.int32)
    disp=np.zeros((N,2),dtype=np.int16)

    for dr in range(-(H-1),H):
        qr0=max(0,-dr); qr1=min(H,H-dr)
        mr0=qr0+dr; mr1=qr1+dr
        if qr1<=qr0: continue
        for dc in range(-(W-1),W):
            qc0=max(0,-dc); qc1=min(W,W-dc)
            mc0=qc0+dc; mc1=qc1+dc
            if qc1<=qc0: continue
            qq=q[:,qr0:qr1,qc0:qc1,:]
            mm=m[:,:,mr0:mr1,mc0:mc1,:]
            eq=np.all(mm==qq[None,...],axis=-1)
            score=eq.sum(axis=(1,2,3),dtype=np.int32)
            improve=score>best
            if np.any(improve):
                best[improve]=score[improve]
                disp[improve,0]=dr
                disp[improve,1]=dc
    return best,disp


def select_coherent_max(
    query:np.ndarray,memory:np.ndarray
)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    score,disp=coherent_displacement_scores(query,memory)
    if len(score)==0:
        return np.empty(0,dtype=np.int64),score,disp
    mx=int(score.max())
    if mx==0:
        return np.empty(0,dtype=np.int64),score,disp
    return np.flatnonzero(score==mx).astype(np.int64),score,disp
