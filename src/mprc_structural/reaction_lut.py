"""Observation-populated ring-medoid ReactionLUT candidate.

Candidate objective (not a frozen MPRC theorem):

For every staple input u in Z256, observations provide a histogram H[u,c] of
center-byte outcomes c. Choose

    L[u] in argmin_y sum_c H[u,c] * cdist(y,c).

Tie breaking is entirely ring-relative:
    1) minimum objective cost;
    2) minimum cdist(y,u);
    3) minimum clockwise displacement (y-u) mod 256.

For an unseen input u, H[u,*]=0 and the neutral choice is L[u]=u.

The rule is integer-only, label-free, and each of the 256 LUT entries is solved
by exhaustive finite MEASURE over exactly 256 candidate outputs.
"""

from __future__ import annotations
import numpy as np

TAU=256

def cdist_table() -> np.ndarray:
    x=np.arange(TAU,dtype=np.int16)
    ab=(x[:,None]-x[None,:])&255
    ba=(x[None,:]-x[:,None])&255
    return np.minimum(ab,ba).astype(np.int16)

CDIST=cdist_table()

def fit_from_counts(counts: np.ndarray) -> np.ndarray:
    h=np.asarray(counts)
    if h.shape!=(256,256):
        raise ValueError("counts must be [256 staple inputs, 256 center states]")
    if np.any(h<0):
        raise ValueError("counts must be non-negative")
    h=h.astype(np.int64,copy=False)

    out=np.empty(256,dtype=np.uint8)
    for u in range(256):
        row=h[u]
        if int(row.sum())==0:
            out[u]=u
            continue

        cost=CDIST.astype(np.int64)@row
        m=int(cost.min())
        cand=np.flatnonzero(cost==m)

        du=CDIST[cand,u].astype(np.int64)
        dmin=int(du.min())
        cand=cand[du==dmin]
        cw=((cand-u)&255).astype(np.int64)
        out[u]=int(cand[int(np.argmin(cw))])
    return out

def observe_counts(staple: np.ndarray, center: np.ndarray) -> np.ndarray:
    u=np.asarray(staple,dtype=np.uint8).reshape(-1)
    c=np.asarray(center,dtype=np.uint8).reshape(-1)
    if len(u)!=len(c):
        raise ValueError("staple and center observations must align")
    code=u.astype(np.int64)*256+c.astype(np.int64)
    return np.bincount(code,minlength=65536).reshape(256,256).astype(np.int64)
