"""v29 candidate IDENTIFY relation for Arshad's ViT.

This module is fixed before benchmark training.

One local observation is the corrected directional ADI-9 byte relation:
    (Lambda, dU1,dU2,dD1,dD2,dF1,dF2,dB1,dB2)

QH4 supplies structural ring roles for those byte values.  IDENTIFY does not
rewrite spatial coordinates and does not collapse the 9-byte relation into a
scalar outside Z256.

For a query/candidate descriptor pair:
  1. give one structural credit when both QH4 clusters have the same J2 type;
  2. accumulate exact circular distance over all 9 ADI bytes.

Across local observations, candidate selection is lexicographic:
  maximize structural credits, then minimize ADI circular energy.

All candidates tied on that exact pair survive to the downstream
BIND -> REACT -> MEASURE stage.  There is no learned weight or threshold.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence

from .qh4 import VACUUM, inverse as qh4_inverse

PATTERN_TYPES=(
    "anchor",
    "vacuum-adjacent",
    "gate-cluster",
    "step-column",
    "cross-quarter",
    "quarter-stripe",
)

def cdist(a:int,b:int)->int:
    a=int(a)&255; b=int(b)&255
    return min((a-b)&255,(b-a)&255)

def qh4_pattern_type(values: Sequence[int]) -> str | None:
    """Integer J2 pattern class for a finite local ring cluster.

    We use the Chapter-15 priority rule on clusters for which every value can
    be handled without inventing a vacuum coordinate. Mixed clusters containing
    a vacuum that are not wholly anchor sets return None and receive no QH4
    structural credit; ADI energy remains available.
    """
    vals=tuple(int(v)&255 for v in values)
    if not vals:
        return None

    # J2 priority 1: anchor set.
    if all((p % 32)==0 for p in vals):
        return "anchor"

    # J2 priority 2: vacuum-adjacent set.
    if all((p % 64) in (1,63) for p in vals):
        return "vacuum-adjacent"

    # Remaining J2 relations require active QH4 addresses.
    if any(p in VACUUM for p in vals):
        return None

    addr=[qh4_inverse(p) for p in vals]
    gamma=[a[0] for a in addr]
    sigma=[a[1] for a in addr]
    theta=[a[2] for a in addr]

    if len(set(gamma))==1:
        return "gate-cluster"
    if len(set(sigma))==1 and len(set(gamma))>1:
        return "step-column"
    if len(set(theta))>=2:
        return "cross-quarter"
    if len(set(theta))==1:
        return "quarter-stripe"
    raise AssertionError("J2 fallback failed")

@dataclass(frozen=True)
class LocalRelation:
    pattern_match: int
    adi_energy: int
    query_type: str | None
    candidate_type: str | None

def local_relation(query_adi:Sequence[int], candidate_adi:Sequence[int]) -> LocalRelation:
    if len(query_adi)!=9 or len(candidate_adi)!=9:
        raise ValueError("IDENTIFY expects ADI-9 descriptors")
    qt=qh4_pattern_type(query_adi)
    ct=qh4_pattern_type(candidate_adi)
    pm=int(qt is not None and qt==ct)
    e=sum(cdist(a,b) for a,b in zip(query_adi,candidate_adi))
    return LocalRelation(pm,e,qt,ct)

@dataclass(frozen=True)
class IdentifyScore:
    pattern_matches: int
    adi_energy: int
    valid_pattern_observations: int

    @property
    def selection_key(self):
        # Smaller tuple is better.
        return (-self.pattern_matches, self.adi_energy)

def identify_score(query:Iterable[Sequence[int]], candidate:Iterable[Sequence[int]]) -> IdentifyScore:
    q=list(query); c=list(candidate)
    if len(q)!=len(c):
        raise ValueError("query and candidate must have aligned local observations")
    matches=0
    energy=0
    valid=0
    for a,b in zip(q,c):
        r=local_relation(a,b)
        matches+=r.pattern_match
        energy+=r.adi_energy
        valid+=int(r.query_type is not None and r.candidate_type is not None)
    return IdentifyScore(matches,energy,valid)

def select_candidate_indices(query, candidates) -> tuple[list[int], list[IdentifyScore]]:
    scores=[identify_score(query,c) for c in candidates]
    if not scores:
        return [],[]
    best=min(s.selection_key for s in scores)
    keep=[i for i,s in enumerate(scores) if s.selection_key==best]
    return keep,scores
