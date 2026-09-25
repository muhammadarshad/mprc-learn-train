"""Local -> Global Arshad Transpose survival gate.

This is the mathematical gate for the Global Attention (GA) probe.

LOCAL
-----
A sample value/state lives in the byte-native local domain:

    P = Z_256.

The 4x4 structural basis has 16 positions and 16^2 = 256 ordered local
relations/states. In MPRC notation this is the completed local unit. We do
not use this file to assert a standard Lie-group identity.

GLOBAL
------
One manifold has

    HV = 128 * 113 = 14,464

globally addressable positions. For N samples/manifolds:

    |G_N| = HV * N.

The globally unique address of manifold n, local manifold position h is

    g = n*HV + h.

ARSHAD TRANSPOSE
----------------
Let

    X : G_N -> Z_256

be the local state stored at each globally unique occurrence.

Transpose is the inverted occurrence map

    X^T[p] = { g in G_N : X(g)=p }.

The 256 buckets are a disjoint partition of G_N. This representation is
implemented as a CSR-like inverted index: offsets[257] + positions[HV*N].
It is NOT a dense 256 x (HV*N) attention matrix.

No classifier, labels, probability, or softmax appear in this gate.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

D=256
SIDE=16
HV=128*113
GEN=7
GEN_INV=183

assert SIDE*SIDE == D
assert SIDE*SIDE - 1 == 255
assert HV == 14_464
assert (GEN*GEN_INV) % D == 1


def global_address(n:int,h:int,N:int)->int:
    if not (0 <= n < N):
        raise ValueError("sample index out of range")
    if not (0 <= h < HV):
        raise ValueError("manifold position out of range")
    return n*HV+h


def decode_global(g:int,N:int)->tuple[int,int]:
    if not (0 <= g < HV*N):
        raise ValueError("global address out of range")
    return divmod(g,HV)


def build_transpose(states:list[int])->tuple[list[int],list[int]]:
    """CSR inverted index: offsets[257], positions[len(states)]."""
    L=len(states)
    counts=[0]*D
    for p in states:
        if not 0 <= int(p) < D:
            raise ValueError("local state outside Z256")
        counts[int(p)] += 1

    offsets=[0]*(D+1)
    for p in range(D):
        offsets[p+1]=offsets[p]+counts[p]
    assert offsets[-1]==L

    positions=[0]*L
    cursor=offsets[:-1].copy()
    for g,p0 in enumerate(states):
        p=int(p0)
        i=cursor[p]
        positions[i]=g
        cursor[p]+=1

    return offsets,positions


def bucket(offsets,positions,p:int)->list[int]:
    a=offsets[p]
    b=offsets[p+1]
    return positions[a:b]


def reconstruct(offsets,positions,L:int)->list[int]:
    out=[None]*L
    for p in range(D):
        for g in bucket(offsets,positions,p):
            assert out[g] is None
            out[g]=p
    assert all(x is not None for x in out)
    return [int(x) for x in out]


report={}

# G1 local construction.
assert SIDE**2 == D
report["G1_local_domain"]={
    "pass":True,
    "basis":"4x4=16",
    "ordered_relation_count":SIDE**2,
    "local_states":D,
}

# G2 generator-7 full local orbit.
for start in range(D):
    seen={((start+GEN*t)&255) for t in range(D)}
    assert len(seen)==D
report["G2_generator7_local_orbit"]={
    "pass":True,
    "starts_tested":D,
    "period":D,
    "inverse_mod256":GEN_INV,
}

# G3 global coordinate bijection over several N.
Ns=[1,2,3,7,16,64]
for N in Ns:
    seen=set()
    for n in range(N):
        for h in range(HV):
            g=global_address(n,h,N)
            assert decode_global(g,N)==(n,h)
            assert g not in seen
            seen.add(g)
    assert len(seen)==HV*N
report["G3_global_address_bijection"]={
    "pass":True,
    "N_values":Ns,
    "HV":HV,
}

# G4 structured transpose partition + roundtrip.
for N in Ns:
    L=HV*N
    states=[g & 255 for g in range(L)]
    offsets,positions=build_transpose(states)
    assert len(offsets)==257
    assert len(positions)==L
    assert reconstruct(offsets,positions,L)==states

    all_g=[]
    for p in range(D):
        b=bucket(offsets,positions,p)
        assert b==sorted(b)
        all_g.extend(b)
    assert len(all_g)==L
    assert len(set(all_g))==L
    assert set(all_g)==set(range(L))
report["G4_structured_transpose"]={
    "pass":True,
    "N_values":Ns,
    "partition_exact":True,
    "roundtrip_exact":True,
}

# G5 random mappings: transpose does not assume uniform occupancy.
rng=random.Random(20260925)
random_trials=[]
for N in (1,2,7,16):
    L=HV*N
    states=[rng.randrange(D) for _ in range(L)]
    offsets,positions=build_transpose(states)
    assert reconstruct(offsets,positions,L)==states
    counts=[offsets[p+1]-offsets[p] for p in range(D)]
    assert sum(counts)==L
    random_trials.append({
        "N":N,
        "global_positions":L,
        "min_bucket":min(counts),
        "max_bucket":max(counts),
    })
report["G5_random_transpose"]={
    "pass":True,
    "trials":random_trials,
}

# G6 same local state, distinct global identities survive.
N=3
L=HV*N
states=[0]*L
for g in (0,1,HV-1,HV,HV+1,2*HV,3*HV-1):
    states[g]=77
offsets,positions=build_transpose(states)
b77=bucket(offsets,positions,77)
assert b77 == [0,1,HV-1,HV,HV+1,2*HV,3*HV-1]
assert len(set(b77))==len(b77)
report["G6_duplicate_local_unique_global"]={
    "pass":True,
    "local_state":77,
    "global_occurrences":b77,
}

# G7 one manifold occupancy arithmetic is deliberately non-uniform under the
# simple round-robin witness: 14464 = 56*256 + 128.
q,r=divmod(HV,D)
assert (q,r)==(56,128)
states=[g&255 for g in range(HV)]
offsets,positions=build_transpose(states)
counts=[offsets[p+1]-offsets[p] for p in range(D)]
assert set(counts)=={56,57}
assert counts.count(57)==128
assert counts.count(56)==128
report["G7_no_uniform_bucket_assumption"]={
    "pass":True,
    "HV_div_256":[q,r],
    "bucket_sizes":[56,57],
    "buckets_each":128,
}

# G8 sparse transpose storage law.
# Abstract element counts, independent of host integer width:
# offsets = 257; positions = HV*N. No 256*(HV*N) dense matrix.
for N in Ns:
    sparse=(D+1)+(HV*N)
    dense=D*(HV*N)
    assert sparse < dense
report["G8_sparse_storage"]={
    "pass":True,
    "representation":"offsets[257] + positions[HV*N]",
    "dense_matrix_required":False,
}

# G9 local/global domains remain separate as N grows.
for N in Ns:
    assert D==256
    assert HV*N >= HV
report["G9_domain_separation"]={
    "pass":True,
    "local_domain":256,
    "global_domain":"14464*N",
}

report["overall"]="ALL PASS"
report["training_gate_open"]=False
report["next"]="Global Attention probe: local query relation -> transpose buckets -> global candidates -> BIND/REACT/MEASURE."
report["claim_boundary"]=(
    "This proves local/global addressing and transpose indexing. It does not yet "
    "define which local states a query should retrieve, nor prove useful attention."
)

root=Path(__file__).resolve().parents[1]
out=root/"results"/"local_global_transpose_survival_gate.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
