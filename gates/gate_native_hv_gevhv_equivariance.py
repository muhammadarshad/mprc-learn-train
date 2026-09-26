"""Native 16x7 <-> 7x16 GEVHV transpose-equivariance gate.

NO TRAINING.

Proves the spatial byte transpose is an exact coordinate transport for
coordinate-wise BIND, five-site REACT with any pointwise LUT, and MEASURE.
"""

import json
from pathlib import Path
import numpy as np

def bind(a,q):
    return ((a.astype(np.uint16)+q.astype(np.uint16))&255).astype(np.uint8)

def react1(x,lut):
    x=np.asarray(x,dtype=np.uint8)
    out=x.copy()
    if x.shape[0]>=3 and x.shape[1]>=3:
        acc=(
            x[1:-1,1:-1].astype(np.uint16)
            +x[:-2,1:-1].astype(np.uint16)
            +x[2:,1:-1].astype(np.uint16)
            +x[1:-1,:-2].astype(np.uint16)
            +x[1:-1,2:].astype(np.uint16)
        )&255
        out[1:-1,1:-1]=lut[acc.astype(np.uint8)]
    return out

def react(x,lut,rounds):
    y=x.copy()
    for _ in range(rounds):
        y=react1(y,lut)
    return y

def measure(a,b):
    aa=a.astype(np.int16);bb=b.astype(np.int16)
    return int(np.minimum((aa-bb)&255,(bb-aa)&255).sum(dtype=np.int64))

R={"gate":"Native H/V GEVHV transpose equivariance","checks":{}}
C=R["checks"]
rng=np.random.default_rng(20260927)

# Arbitrary LUTs, not only identity.
luts=[
    np.arange(256,dtype=np.uint8),
    rng.permutation(256).astype(np.uint8),
    ((np.arange(256,dtype=np.uint16)*73+19)&255).astype(np.uint8),
]

cases=0
for trial in range(64):
    a=rng.integers(0,256,size=(16,7),dtype=np.uint8)
    q=rng.integers(0,256,size=(16,7),dtype=np.uint8)
    at=a.T.copy(); qt=q.T.copy()

    assert np.array_equal(bind(a,q).T,bind(at,qt))
    assert measure(a,q)==measure(at,qt)

    for lut in luts:
        for rounds in (1,2,7):
            ra=react(bind(a,q),lut,rounds)
            rt=react(bind(at,qt),lut,rounds)
            assert np.array_equal(ra.T,rt)
            assert measure(ra,q)==measure(rt,qt)
            cases+=1

C["random_full_block_cases"]={"pass":True,"cases":cases,"arbitrary_LUTs":len(luts)}

# Exhaustive single-site/value family on all 112 native positions.
basis=0
identity=np.arange(256,dtype=np.uint8)
for r in range(16):
    for c in range(7):
        for v in range(256):
            a=np.zeros((16,7),dtype=np.uint8)
            q=np.zeros((16,7),dtype=np.uint8)
            a[r,c]=v
            q[(r+3)%16,(c+2)%7]=(v*37+11)&255
            assert np.array_equal(bind(a,q).T,bind(a.T.copy(),q.T.copy()))
            ra=react(bind(a,q),identity,1)
            rt=react(bind(a.T.copy(),q.T.copy()),identity,1)
            assert np.array_equal(ra.T,rt)
            assert measure(ra,q)==measure(rt,q.T.copy())
            basis+=1
C["single_site_value_cases"]={"pass":True,"cases":basis}

R["status"]="PASS"
R["theorem"]=(
    "For the native rectangular spatial transpose T, coordinate-wise Z256 BIND commutes "
    "with T; the symmetric five-site S5 stencil and pointwise LUT make REACT commute "
    "with T; circular-distance MEASURE is invariant under coordinate permutation. "
    "Therefore an H/V phase change is exact transport when state and query are "
    "transposed together."
)
R["claim_boundary"]=(
    "This closes H/V equivariance of the GEVHV core. It does not choose the semantic "
    "schedule for QH4/directional IDENTIFY, and it does not identify spatial transpose "
    "with Paper3 T_k."
)
root=Path(__file__).resolve().parents[1]
out=root/"results"/"native_hv_gevhv_equivariance.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
