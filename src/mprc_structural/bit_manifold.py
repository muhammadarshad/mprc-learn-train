"""OPTIONAL physical packing/accounting utility — NOT the canonical ViT manifold.

This module was introduced while correcting a byte-vs-pixel/storage-accounting
mistake.  Its 128x113 bit packing view MUST NOT be used to redefine the frozen
Arshad-ViT computational manifold or its operators.

Frozen ViT computation remains:
    computational manifold : 128 x 113 Z256 state/address slots
    execution              : two 64 x 113 slabs
    transport              : q_t = q_0 + 7*t (mod 64)
    attention              : IDENTIFY -> BIND -> REACT -> MEASURE

The helpers below only show that an unrelated 16x113 byte buffer can be packed
into 128x113 physical bits.  They are useful for storage/accounting experiments,
not as a replacement execution geometry.

Do not import this module from canonical attention, routing, REACT, SELECT, or
training code.
"""
from __future__ import annotations
import numpy as np

BIT_H=128
SAMPLES=113
CHANNELS=16
BITS_PER_BYTE=8
MANIFOLD_BITS=BIT_H*SAMPLES
STORAGE_BYTES=MANIFOLD_BITS//8
BYTE_VIEW_BYTES=CHANNELS*SAMPLES

assert BIT_H==CHANNELS*BITS_PER_BYTE
assert MANIFOLD_BITS==14_464
assert STORAGE_BYTES==1_808
assert BYTE_VIEW_BYTES==1_808


def pack_bytes(byte_view:np.ndarray)->np.ndarray:
    x=np.asarray(byte_view)
    if x.shape!=(CHANNELS,SAMPLES) or x.dtype!=np.uint8:
        raise ValueError("expected uint8 [16,113]")
    out=np.empty((BIT_H,SAMPLES),dtype=np.uint8)
    for c in range(CHANNELS):
        v=x[c]
        for b in range(BITS_PER_BYTE):
            out[BITS_PER_BYTE*c+b]=((v>>b)&1).astype(np.uint8)
    return out


def unpack_bytes(bits:np.ndarray)->np.ndarray:
    x=np.asarray(bits)
    if x.shape!=(BIT_H,SAMPLES) or x.dtype!=np.uint8:
        raise ValueError("expected uint8 [128,113] bit manifold")
    if np.any((x!=0)&(x!=1)):
        raise ValueError("bit manifold values must be 0 or 1")
    out=np.zeros((CHANNELS,SAMPLES),dtype=np.uint8)
    for c in range(CHANNELS):
        v=np.zeros(SAMPLES,dtype=np.uint16)
        for b in range(BITS_PER_BYTE):
            v |= x[BITS_PER_BYTE*c+b].astype(np.uint16)<<b
        out[c]=v.astype(np.uint8)
    return out


def storage_counts()->dict[str,int]:
    return {
        "bit_rows":BIT_H,
        "samples":SAMPLES,
        "channels":CHANNELS,
        "bits_per_byte":BITS_PER_BYTE,
        "manifold_bits":MANIFOLD_BITS,
        "storage_bytes":STORAGE_BYTES,
        "byte_view_bytes":BYTE_VIEW_BYTES,
    }
