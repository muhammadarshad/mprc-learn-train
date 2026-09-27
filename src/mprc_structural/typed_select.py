"""Exact typed-field SELECT for the survived ADI-9 routing candidate.

For fixed typed slots f=0..F-1, every descriptor is a full 9-byte Z256 ADI state.

    M_n(Q) = sum_f 1[Q_f == Memory[n,f]]

SELECT keeps every memory sample attaining max_n M_n.

There is no threshold, Top-K, learned weight, radius or label use.
If the maximum score is zero, SELECT returns no candidate (abstention).
"""

from __future__ import annotations
import numpy as np


def exact_match_counts(query: np.ndarray, memory: np.ndarray) -> np.ndarray:
    q=np.asarray(query)
    m=np.asarray(memory)
    if q.ndim!=2 or q.shape[1]!=9:
        raise ValueError("query must be [F,9]")
    if m.ndim!=3 or m.shape[1:]!=q.shape:
        raise ValueError("memory must be [N,F,9]")
    if q.dtype!=np.uint8 or m.dtype!=np.uint8:
        raise TypeError("descriptors must be uint8")
    return np.all(m==q[None,:,:],axis=2).sum(axis=1,dtype=np.int32)


def select_exact_max(query: np.ndarray, memory: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    counts=exact_match_counts(query,memory)
    if len(counts)==0:
        return np.empty(0,dtype=np.int64),counts
    best=int(counts.max())
    if best==0:
        return np.empty(0,dtype=np.int64),counts
    return np.flatnonzero(counts==best).astype(np.int64),counts
