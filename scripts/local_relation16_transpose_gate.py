"""4x4 / 16-state local relation + transpose survival gate.

NO TRAINING. NO LABELS.

The corrected local attention domain has 16 basis positions (4x4), with
ordered relation space

    R16 = B x B,  |B|=16, |R16|=256.

Encode:
    p(i,j) = 16*i + j  in 0..255.

Local transpose reverses relation orientation:
    tau(i,j) = (j,i)
    tau(p) = 16*(p mod 16) + floor(p/16).

This is an exact involution on all 256 local states.

The generator-7 walk remains a permutation of the same local 256-state domain.
This gate does not claim a standard SU(16) group identity; it validates only
the 16x16=256 relation arithmetic and transpose operator needed by attention.
"""

from __future__ import annotations
import json
from pathlib import Path

SIDE=16
D=256
GEN=7
GEN_INV=183

def enc(i:int,j:int)->int:
    if not (0<=i<SIDE and 0<=j<SIDE):
        raise ValueError
    return SIDE*i+j

def dec(p:int)->tuple[int,int]:
    if not 0<=p<D:
        raise ValueError
    return divmod(p,SIDE)

def transpose(p:int)->int:
    i,j=dec(p)
    return enc(j,i)

report={}

# G1 exact 16x16<->256 bijection.
seen=set()
for i in range(SIDE):
    for j in range(SIDE):
        p=enc(i,j)
        assert dec(p)==(i,j)
        assert p not in seen
        seen.add(p)
assert seen==set(range(D))
report["G1_relation_bijection"]={"pass":True,"basis":16,"ordered_relations":256}

# G2 transpose involution exhaustive.
for p in range(D):
    assert transpose(transpose(p))==p
report["G2_transpose_involution"]={"pass":True,"states":256}

# G3 orbit decomposition: 16 diagonal fixed points + 120 reversed pairs.
fixed=[p for p in range(D) if transpose(p)==p]
pairs=[]
used=set(fixed)
for p in range(D):
    if p in used:
        continue
    q=transpose(p)
    assert q!=p
    pairs.append((p,q))
    used.add(p);used.add(q)
assert len(fixed)==16
assert len(pairs)==120
assert len(used)==256
report["G3_orbits"]={
    "pass":True,
    "fixed_diagonal_relations":len(fixed),
    "transpose_pairs":len(pairs),
}

# G4 transpose preserves unordered endpoints and reverses orientation.
for p in range(D):
    i,j=dec(p)
    q=transpose(p)
    a,b=dec(q)
    assert (a,b)==(j,i)
    assert {a,b}=={i,j}
report["G4_orientation_reversal"]={"pass":True}

# G5 generator-7 full orbit and inverse.
assert (GEN*GEN_INV)%D==1
for start in range(D):
    orbit=[(start+GEN*t)&255 for t in range(D)]
    assert len(set(orbit))==D
report["G5_generator7"]={
    "pass":True,
    "period":256,
    "inverse_mod256":GEN_INV,
}

# G6 transpose remains an involution when expressed in generator-orbit coordinates.
# Let phi_s(t)=s+7t. Conjugate tau by phi_s.
for start in range(D):
    def phi(t): return (start+GEN*t)&255
    def phi_inv(p): return ((p-start)*GEN_INV)&255
    for t in range(D):
        T=lambda x: phi_inv(transpose(phi(x)))
        assert T(T(t))==t
report["G6_generator_conjugated_transpose"]={
    "pass":True,
    "starts":256,
    "states_per_start":256,
}

# G7 relation transpose is not assumed to be +constant on Z256.
# Record actual displacement distribution to prevent future collapse to a fake scalar shift.
deltas={}
for p in range(D):
    d=(transpose(p)-p)&255
    deltas[d]=deltas.get(d,0)+1
assert sum(deltas.values())==256
report["G7_not_single_shift"]={
    "pass":True,
    "distinct_mod256_displacements":len(deltas),
    "displacement_histogram":{str(k):v for k,v in sorted(deltas.items())},
}

report["overall"]="ALL PASS"
report["candidate_attention_policy"]="retrieve p and tau(p), preserving each global occurrence separately"
report["claim_boundary"]=(
    "This proves the 16x16 local relation encoding and transpose involution. "
    "It does not prove that transpose-pair retrieval is sufficient for learned attention."
)

root=Path(__file__).resolve().parents[1]
out=root/"results"/"local_relation16_transpose_gate.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
