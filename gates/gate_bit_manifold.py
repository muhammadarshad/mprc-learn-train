"""Exact storage/information gate for corrected 128x113 bit manifold."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from mprc_structural.bit_manifold import *

rng=np.random.default_rng(20260927)
R={"gate":"Arshad-ViT 14,464-bit storage correction","checks":{}}
C=R["checks"]

assert MANIFOLD_BITS==14_464
assert STORAGE_BYTES==1_808
assert BYTE_VIEW_BYTES==16*113==1_808
C["G1_capacity"]={"pass":True,"bits":MANIFOLD_BITS,"bytes":STORAGE_BYTES,"byte_view":[16,113]}

for i in range(512):
    x=rng.integers(0,256,size=(16,113),dtype=np.uint8)
    b=pack_bytes(x)
    y=unpack_bytes(b)
    assert b.shape==(128,113)
    assert set(np.unique(b)).issubset({0,1})
    assert np.array_equal(x,y)
C["G2_roundtrip"]={"pass":True,"random_byte_views":512}

# Every possible byte at multiple channel/sample addresses.
cases=0
for c,s in ((0,0),(0,112),(7,56),(15,0),(15,112)):
    for v in range(256):
        x=np.zeros((16,113),dtype=np.uint8); x[c,s]=v
        assert unpack_bytes(pack_bytes(x))[c,s]==v
        cases+=1
C["G3_all_byte_values"]={"pass":True,"cases":cases}

# One byte occupies exactly eight storage bits at one sample column.
x=np.zeros((16,113),dtype=np.uint8); x[5,77]=0xA5
b=pack_bytes(x)
nz=np.argwhere(b!=0)
assert all(int(col)==77 for row,col in nz)
assert all(40<=int(row)<48 for row,col in nz)
C["G4_no_storage_alias"]={"pass":True,"channel":5,"sample":77,"bit_rows":[40,47]}

R["status"]="PASS";R["all_pass"]=all(v["pass"] for v in C.values())
R["frozen_distinction"]={
 "source_pixels":"task-dependent unique image coordinates",
 "source_bytes":"actual byte-valued information read from those pixels/channels",
 "byte_view":"at most 1,808 Z256 bytes",
 "storage":"14,464 physical bits",
 "computational_states":"must be reported separately; never called pixels or storage bytes"
}
out=Path("results/bit_manifold_survival.json");out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
