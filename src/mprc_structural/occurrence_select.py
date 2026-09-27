"""Global-occurrence exact SELECT for typed directional descriptors.

For each sample, the local descriptor field is treated as a multiset over
D in Z256^9.  Query-memory support is exact multiset intersection:

    M_n(Q) = sum_D min(c_Q(D), c_n(D))

This is invariant to permutation/displacement of local occurrence positions.
All maximum-support samples survive.  If max support is zero, abstain.

Global occurrence positions are not collapsed by this scoring definition; the
routing/index layer may retain them for downstream context.
"""

from __future__ import annotations
import numpy as np

VOID9=np.dtype((np.void,9))


def _flat_keys(x:np.ndarray)->np.ndarray:
    a=np.ascontiguousarray(x,dtype=np.uint8)
    if a.ndim<2 or a.shape[-1]!=9:
        raise ValueError("descriptor field must end in 9 bytes")
    return a.reshape(-1,9).view(VOID9).reshape(-1)


def multiset_match_counts(query:np.ndarray,memory:np.ndarray)->np.ndarray:
    q=np.asarray(query)
    m=np.asarray(memory)
    if q.ndim!=2 or q.shape[1]!=9 or q.dtype!=np.uint8:
        raise ValueError("query must be uint8 [A,9]")
    if m.ndim!=3 or m.shape[2]!=9 or m.dtype!=np.uint8:
        raise ValueError("memory must be uint8 [N,A,9]")
    N=m.shape[0]
    score=np.zeros(N,dtype=np.int32)

    qk,qc=np.unique(_flat_keys(q),return_counts=True)
    for key,qcount in zip(qk,qc):
        # Reference implementation deliberately simple.
        for n in range(N):
            mk=_flat_keys(m[n])
            c=int(np.count_nonzero(mk==key))
            if c:
                score[n]+=min(int(qcount),c)
    return score


def select_multiset_max(query:np.ndarray,memory:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    score=multiset_match_counts(query,memory)
    if len(score)==0:
        return np.empty(0,dtype=np.int64),score
    best=int(score.max())
    if best==0:
        return np.empty(0,dtype=np.int64),score
    return np.flatnonzero(score==best).astype(np.int64),score


class SortedOccurrenceSelectIndex:
    """Exact sparse index equivalent to multiset_match_counts."""

    def __init__(self,memory:np.ndarray):
        m=np.asarray(memory)
        if m.ndim!=3 or m.shape[2]!=9 or m.dtype!=np.uint8:
            raise ValueError("memory must be uint8 [N,A,9]")
        self.n,self.a=m.shape[:2]
        keys=_flat_keys(m)
        samples=np.repeat(np.arange(self.n,dtype=np.int32),self.a)
        order=np.argsort(keys,kind="stable")
        self.keys=keys[order].copy()
        self.samples=samples[order].copy()

    def match_counts(self,query:np.ndarray)->np.ndarray:
        q=np.asarray(query)
        if q.ndim!=2 or q.shape[1]!=9 or q.dtype!=np.uint8:
            raise ValueError("query must be uint8 [A,9]")
        score=np.zeros(self.n,dtype=np.int32)
        qk,qc=np.unique(_flat_keys(q),return_counts=True)
        for key,qcount in zip(qk,qc):
            lo=int(np.searchsorted(self.keys,key,side="left"))
            hi=int(np.searchsorted(self.keys,key,side="right"))
            if hi<=lo:
                continue
            ids=self.samples[lo:hi]
            uid,mc=np.unique(ids,return_counts=True)
            score[uid]+=np.minimum(int(qcount),mc).astype(np.int32)
        return score

    def select(self,query:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        score=self.match_counts(query)
        if self.n==0:
            return np.empty(0,dtype=np.int64),score
        best=int(score.max())
        if best==0:
            return np.empty(0,dtype=np.int64),score
        return np.flatnonzero(score==best).astype(np.int64),score
