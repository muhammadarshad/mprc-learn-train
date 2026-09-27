"""Exact gate for corrected byte-state MPRC attention."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from mprc_structural.bit_manifold import pack_bytes,unpack_bytes,STORAGE_BYTES,MANIFOLD_BITS
from mprc_structural.byte_attention import *

rng=np.random.default_rng(20260927)
R={"gate":"corrected 1,808-byte BIND-REACT-MEASURE","checks":{}}
C=R["checks"]

assert ORDER16.tolist()==[(7*t)&15 for t in range(16)]
assert len(set(map(int,ORDER16)))==16
C["G1_generator7_Z16"]={"pass":True,"orbit":ORDER16.tolist(),"inverse":7}

for _ in range(512):
    x=rng.integers(0,256,size=(16,113),dtype=np.uint8)
    assert np.array_equal(inverse_transport(transport(x)),x)
C["G2_transport_inverse"]={"pass":True,"cases":512}

# Storage is lossless but not an arithmetic state.
for _ in range(128):
    a=rng.integers(0,256,size=(16,113),dtype=np.uint8)
    q=rng.integers(0,256,size=(16,113),dtype=np.uint8)
    o0=forward(a,q,rounds=3)
    o1=forward_stored_bits(pack_bytes(a),pack_bytes(q),rounds=3)
    assert o0["energy"]==o1["energy"]
    assert np.array_equal(o0["state"],unpack_bytes(o1["state_bits"]))
C["G3_storage_execution_equivalence"]={"pass":True,"cases":128,
 "storage_bits":MANIFOLD_BITS,"storage_bytes":STORAGE_BYTES}

# Exact Z256 BIND and circular MEASURE.
for _ in range(256):
    a=rng.integers(0,256,size=(16,113),dtype=np.uint8)
    b=rng.integers(0,256,size=(16,113),dtype=np.uint8)
    z=bind(a,b)
    assert z.dtype==np.uint8
    assert measure(a,b)==measure(b,a)
    assert measure(a,a)==0
C["G4_ring_laws"]={"pass":True,"cases":256}

# One impulse: report COMPUTATIONAL byte support only; never call it pixels.
x=np.zeros((16,113),dtype=np.uint8);x[8,56]=1
counts=[]
state=x
for r in range(1,8):
    state=react_once(state)
    counts.append(int(np.count_nonzero(state)))
C["G5_react_computational_support"]={"pass":True,"rounds":7,"byte_state_nonzero_counts":counts,
 "meaning":"computational byte states, NOT source pixels"}

R["status"]="PASS";R["all_pass"]=all(v["pass"] for v in C.values())
R["frozen"]={
 "storage":"128x113 bits = 14,464 bits = 1,808 bytes",
 "computation":"16x113 Z256 bytes = 1,808 byte states",
 "generator":"7 on Z16 byte rows",
 "prohibition":"never apply Z256 arithmetic directly to 128 storage-bit rows"
}
out=Path("results/byte_attention_survival.json");out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
