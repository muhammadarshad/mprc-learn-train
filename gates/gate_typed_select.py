"""No-dataset survival gate for typed exact-match SELECT.

NO DATASET. NO LABELS. NO TRAINING.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from mprc_structural.directional_adi import encode as adi_encode, transpose as adi_transpose
from mprc_structural.typed_select import exact_match_counts, select_exact_max

R={"gate":"typed exact descriptor field SELECT","checks":{}}
C=R["checks"]

rng=np.random.default_rng(20260927)
F=37
N=12

# Deterministic full-byte directional fields.
raw=rng.integers(0,256,size=(N,F,9),dtype=np.uint8)
mem=np.empty_like(raw)
for n in range(N):
    for f in range(F):
        mem[n,f]=np.asarray(adi_encode(raw[n,f]),dtype=np.uint8)

# G1 self identity must be maximal and exact.
self_cases=0
for n in range(N):
    sel,c=select_exact_max(mem[n],mem)
    assert n in set(map(int,sel))
    assert int(c[n])==F
    assert int(c.max())==F
    self_cases+=1
C["G1_self_maximal"]={"pass":True,"cases":self_cases,"features":F}

# G2 one full descriptor mutation reduces exactly one match.
mutation_cases=0
for n in range(N):
    for f in range(F):
        q=mem[n].copy()
        q[f,0]=(int(q[f,0])+1)&255
        c=exact_match_counts(q,mem)
        assert int(c[n])==F-1
        mutation_cases+=1
C["G2_one_descriptor_mutation"]={"pass":True,"cases":mutation_cases}

# G3 each of the nine ADI coordinates is decision relevant.
coord_cases=0
for j in range(9):
    q=mem[0].copy()
    q[3,j]=(int(q[3,j])+1)&255
    c=exact_match_counts(q,mem)
    assert int(c[0])==F-1
    coord_cases+=1
C["G3_all_ADI_coordinates"]={"pass":True,"cases":coord_cases}

# G4 common permutation of typed slots leaves all scores unchanged.
q=mem[2].copy()
c0=exact_match_counts(q,mem)
for _ in range(256):
    p=rng.permutation(F)
    c1=exact_match_counts(q[p],mem[:,p,:])
    assert np.array_equal(c0,c1)
C["G4_common_slot_permutation"]={"pass":True,"cases":256}

# G5 H/V canonicalization: common spatial transpose of descriptors leaves scores.
# directional_adi.transpose is a coordinate permutation on each descriptor.
def tfield(x):
    y=np.empty_like(x)
    for i,z in enumerate(x):
        y[i]=np.asarray(adi_transpose(tuple(map(int,z))),dtype=np.uint8)
    return y

q=mem[5]
mT=np.empty_like(mem)
for n in range(N):
    mT[n]=tfield(mem[n])
c0=exact_match_counts(q,mem)
c1=exact_match_counts(tfield(q),mT)
assert np.array_equal(c0,c1)
C["G5_HV_canonical_score"]={"pass":True}

# G6 exact duplicate memories must remain tied; no hidden tie break.
m2=np.concatenate([mem,mem[[0]]],axis=0)
sel,c=select_exact_max(mem[0],m2)
assert set(map(int,sel))=={0,N}
assert int(c[0])==int(c[N])==F
C["G6_exact_ties_survive"]={"pass":True,"selected":list(map(int,sel))}

# G7 no-evidence case must abstain, not pick arbitrary memory.
q=np.bitwise_xor(mem[0],np.uint8(0xFF))
# Guarantee no accidental exact descriptor by replace until none.
while int(exact_match_counts(q,mem).max())!=0:
    q=(q+np.uint8(1)).astype(np.uint8)
sel,c=select_exact_max(q,mem)
assert len(sel)==0 and int(c.max())==0
C["G7_zero_evidence_abstains"]={"pass":True}

R["status"]="PASS"
R["all_pass"]=all(x["pass"] for x in C.values())
R["rule"]="SELECT all memories with maximal count of exact full ADI-9 descriptor matches; abstain if max count is zero."
R["prohibitions"]={"top_k":False,"threshold":False,"radius":False,"learned_weights":False,"labels":False}
R["training_authorization"]="SELECT SURVIVED. It may now be used after the survived exact typed IDENTIFY route and before BIND->REACT->MEASURE. Recognition usefulness is empirical."

root=Path(__file__).resolve().parents[1]
out=root/"results"/"typed_select_survival.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
