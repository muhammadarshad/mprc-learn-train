"""v28 exact typed IDENTIFY: QH4 tag applied to directional ADI-9 values.

This closes only the type/interface composition:
  directional bytes -> ADI9 bytes -> QH4 tags.

It does NOT define a routing/state-transition action.
"""

from __future__ import annotations
from dataclasses import dataclass
from .directional_adi import encode as adi_encode, decode as adi_decode
from .qh4_mixed import VACUUM, decode_address, active_address

@dataclass(frozen=True)
class VacuumTag:
    value:int

@dataclass(frozen=True)
class ActiveTag:
    theta:int
    a:int
    p:int
    sigma:int

def qh4_tag(value:int):
    v=int(value)&0xFF
    if v in VACUUM:
        return VacuumTag(v)
    th,a,p,s=decode_address(v)
    return ActiveTag(th,a,p,s)

def qh4_untag(tag)->int:
    if isinstance(tag,VacuumTag):
        v=int(tag.value)&0xFF
        if v not in VACUUM:
            raise ValueError("invalid vacuum tag")
        return v
    if isinstance(tag,ActiveTag):
        return active_address(tag.theta,tag.a,tag.p,tag.sigma)
    raise TypeError(type(tag))

def identify_directional(values9):
    adi=tuple(adi_encode(values9))
    tags=tuple(qh4_tag(v) for v in adi)
    return adi,tags

def recover_directional(adi,tags):
    a=tuple(int(v)&0xFF for v in adi)
    if len(a)!=9 or len(tags)!=9:
        raise ValueError("expected 9 ADI values and 9 QH4 tags")
    rebuilt=tuple(qh4_untag(t) for t in tags)
    if rebuilt!=a:
        raise ValueError("tag/value mismatch")
    return tuple(adi_decode(a))
