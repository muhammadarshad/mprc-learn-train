"""MPRC attention on the corrected 1,808-byte computational state.

Storage:
    128 x 113 = 14,464 bits.

Computation:
    16 x 113 = 1,808 Z256 bytes.

The two views are exactly related by bit_manifold.pack_bytes/unpack_bytes.
Z256 BIND/REACT/MEASURE operate ONLY on the byte view.  The 128 storage rows
are bits and must never be treated as 128 independent Z256 byte states.

Generator transport on the 16 byte rows:
    q_t = q_0 + 7 t (mod 16)
which is a full orbit because gcd(7,16)=1.

REACT is the existing five-site cross, now applied to the actual byte-state
geometry. Boundaries remain identity.
"""
from __future__ import annotations
import numpy as np
from .bit_manifold import CHANNELS,SAMPLES,pack_bytes,unpack_bytes

GEN=7
ROWS=CHANNELS
COLS=SAMPLES
assert ROWS==16 and COLS==113
assert pow(GEN,-1,ROWS)==7  # 7*7=49=1 mod16

ORDER16=np.asarray([(GEN*t)&15 for t in range(16)],dtype=np.int64)
assert len(set(map(int,ORDER16)))==16

def validate_byte_state(x):
    a=np.asarray(x)
    if a.shape!=(ROWS,COLS) or a.dtype!=np.uint8:
        raise ValueError("expected uint8 [16,113]")
    return a

def transport(state:np.ndarray)->np.ndarray:
    x=validate_byte_state(state)
    return x[ORDER16,:].copy()

def inverse_transport(logical:np.ndarray)->np.ndarray:
    x=validate_byte_state(logical)
    out=np.empty_like(x);out[ORDER16,:]=x
    return out

def bind(a:np.ndarray,b:np.ndarray)->np.ndarray:
    x=validate_byte_state(a);y=validate_byte_state(b)
    return ((x.astype(np.uint16)+y.astype(np.uint16))&255).astype(np.uint8)

def validate_lut(lut):
    if lut is None:return np.arange(256,dtype=np.uint8)
    z=np.asarray(lut)
    if z.shape!=(256,) or z.dtype!=np.uint8:raise ValueError("lut must be uint8[256]")
    return z

def react_once(logical:np.ndarray,lut=None)->np.ndarray:
    x=validate_byte_state(logical);L=validate_lut(lut)
    out=x.copy()
    acc=(x[1:-1,1:-1].astype(np.uint16)
         +x[:-2,1:-1].astype(np.uint16)
         +x[2:,1:-1].astype(np.uint16)
         +x[1:-1,:-2].astype(np.uint16)
         +x[1:-1,2:].astype(np.uint16))&255
    out[1:-1,1:-1]=L[acc.astype(np.uint8)]
    return out

def react(logical:np.ndarray,lut=None,rounds:int=1)->np.ndarray:
    x=validate_byte_state(logical).copy()
    if int(rounds)<1:raise ValueError("rounds >=1")
    for _ in range(int(rounds)):x=react_once(x,lut)
    return x

def measure(a:np.ndarray,b:np.ndarray)->int:
    x=validate_byte_state(a).astype(np.int16)
    y=validate_byte_state(b).astype(np.int16)
    ab=(x-y)&255;ba=(y-x)&255
    return int(np.minimum(ab,ba).sum(dtype=np.int64))

def forward(state:np.ndarray,query:np.ndarray,*,lut=None,rounds:int=1)->dict:
    ts=transport(state);tq=transport(query)
    z=bind(ts,tq)
    r=react(z,lut=lut,rounds=rounds)
    return {"energy":measure(r,tq),"logical_state":r,"state":inverse_transport(r),
            "rounds":int(rounds),"generator":GEN}

def forward_stored_bits(state_bits:np.ndarray,query_bits:np.ndarray,*,lut=None,rounds:int=1)->dict:
    s=unpack_bytes(state_bits);q=unpack_bytes(query_bits)
    out=forward(s,q,lut=lut,rounds=rounds)
    out["state_bits"]=pack_bytes(out["state"])
    return out
