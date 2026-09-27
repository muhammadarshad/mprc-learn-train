"""Sorted exact index for typed ADI-9 field SELECT.

This is an optimization only.  It must produce exactly the same per-memory
match counts and selected maxima as typed_select.select_exact_max.
"""

from __future__ import annotations
import numpy as np

VOID9=np.dtype((np.void,9))


def _keys9(x:np.ndarray)->np.ndarray:
    a=np.ascontiguousarray(x,dtype=np.uint8)
    if a.shape[-1]!=9:
        raise ValueError("last dimension must be 9")
    return a.reshape(-1,9).view(VOID9).reshape(a.shape[:-1])


class SortedTypedSelectIndex:
    def __init__(self,memory:np.ndarray):
        m=np.asarray(memory)
        if m.ndim!=3 or m.shape[2]!=9 or m.dtype!=np.uint8:
            raise ValueError("memory must be uint8 [N,F,9]")
        self.n,self.f=m.shape[:2]
        self.keys=[]
        self.samples=[]
        for j in range(self.f):
            k=_keys9(m[:,j,:]).reshape(self.n)
            order=np.argsort(k,kind="stable")
            self.keys.append(k[order].copy())
            self.samples.append(order.astype(np.int32,copy=False))

    def match_counts(self,query:np.ndarray)->np.ndarray:
        q=np.asarray(query)
        if q.shape!=(self.f,9) or q.dtype!=np.uint8:
            raise ValueError("query must be uint8 [F,9]")
        qk=_keys9(q).reshape(self.f)
        counts=np.zeros(self.n,dtype=np.int32)
        for j in range(self.f):
            keys=self.keys[j]
            key=qk[j]
            lo=int(np.searchsorted(keys,key,side="left"))
            hi=int(np.searchsorted(keys,key,side="right"))
            if hi>lo:
                counts[self.samples[j][lo:hi]]+=1
        return counts

    def select(self,query:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        counts=self.match_counts(query)
        if self.n==0:
            return np.empty(0,dtype=np.int64),counts
        best=int(counts.max())
        if best==0:
            return np.empty(0,dtype=np.int64),counts
        return np.flatnonzero(counts==best).astype(np.int64),counts
