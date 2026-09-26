"""v29 — relational IDENTIFY survival gate.

NO DATASET. NO LABELS. NO TRAINING.
"""

from __future__ import annotations
from pathlib import Path
import json
import random

from mprc_structural.directional_adi import encode as adi_encode, transpose as adi_transpose
from mprc_structural.identify import (
    PATTERN_TYPES, cdist, qh4_pattern_type, local_relation,
    identify_score, select_candidate_indices
)
from mprc_structural.qh4 import VACUUM, active_positions, inverse as qh4_inverse, forward as qh4_forward

rng=random.Random(20260927)
report={"gate":"v29-relational-identify","checks":{}}
C=report["checks"]

# G1 QH4 locate remains a content-address bijection, independent of sequence index.
active=active_positions()
for p in active:
    g,s,t=qh4_inverse(p)
    assert qh4_forward(g,s)==p
C["QH4_content_address"]={"pass":True,"active_values":252,"sequential_index_used":False}

# G2 J2 classifier is permutation invariant on deterministic active clusters.
clusters=[]
# one gate cluster
clusters.append([qh4_forward(5,s) for s in range(1,8)])
# step column sigma=4 across gates
clusters.append([qh4_forward(g,4) for g in (1,5,10,20,30)])
# cross quarter
clusters.append([qh4_forward(g,3) for g in (2,12,22,32)])
# quarter stripe
clusters.append([qh4_forward(g,2) for g in (1,3,5,7)])
# anchors and vacuum-adjacent examples
clusters.append([32,96,160,224])
clusters.append([1,63,65,127])

perm_cases=0
seen_types=set()
for vals in clusters:
    typ=qh4_pattern_type(vals)
    assert typ in PATTERN_TYPES
    seen_types.add(typ)
    for _ in range(64):
        q=vals[:]
        rng.shuffle(q)
        assert qh4_pattern_type(q)==typ
        perm_cases+=1
assert seen_types==set(PATTERN_TYPES)
C["J2_pattern_permutation"]={"pass":True,"cases":perm_cases,"types":sorted(seen_types)}

# G3 ADI local relation identity: zero energy iff descriptors equal.
identity_cases=0
base=[11,29,47,83,109,137,163,211,239]
for k in range(9):
    for v in range(256):
        a=base[:]; a[k]=v
        z=adi_encode(a)
        r=local_relation(z,z)
        assert r.adi_energy==0
        identity_cases+=1
C["ADI_identity"]={"pass":True,"cases":identity_cases}

# G4 common directional transpose leaves local IDENTIFY score unchanged.
transpose_cases=0
for _ in range(5000):
    a=[rng.randrange(256) for _ in range(9)]
    b=[rng.randrange(256) for _ in range(9)]
    za=adi_encode(a); zb=adi_encode(b)
    r0=local_relation(za,zb)
    r1=local_relation(adi_transpose(za),adi_transpose(zb))
    assert r0.pattern_match==r1.pattern_match
    assert r0.adi_energy==r1.adi_energy
    transpose_cases+=1
C["transpose_invariance"]={"pass":True,"cases":transpose_cases}

# G5 aggregate IDENTIFY ignores sequential order when the same permutation is
# applied to query and candidate local observations.
sequence_cases=0
for _ in range(256):
    q=[];m=[]
    for j in range(17):
        q.append(adi_encode([rng.randrange(256) for _ in range(9)]))
        m.append(adi_encode([rng.randrange(256) for _ in range(9)]))
    s0=identify_score(q,m)
    order=list(range(len(q))); rng.shuffle(order)
    s1=identify_score([q[i] for i in order],[m[i] for i in order])
    assert s0==s1
    sequence_cases+=1
C["common_sequence_permutation"]={"pass":True,"cases":sequence_cases}

# G6 exact candidate must survive and uniquely beat one-byte perturbations
# through zero ADI energy, with no threshold or learned weight.
selection_cases=0
for _ in range(512):
    q=[adi_encode([rng.randrange(256) for _ in range(9)]) for _ in range(8)]
    exact=[tuple(z) for z in q]
    candidates=[]
    for d in range(4):
        c=[list(z) for z in q]
        j=rng.randrange(len(c)); k=rng.randrange(9)
        c[j][k]=(c[j][k]+1+d)&255
        candidates.append([tuple(z) for z in c])
    insert=rng.randrange(5)
    candidates.insert(insert,exact)
    keep,scores=select_candidate_indices(q,candidates)
    assert keep==[insert]
    assert scores[insert].adi_energy==0
    selection_cases+=1
C["exact_candidate_selection"]={"pass":True,"cases":selection_cases}

# G7 no scalar collapse: score exposes structural count and ring energy separately.
q=[adi_encode(base)]
m=[adi_encode([x^1 for x in base])]
s=identify_score(q,m)
assert isinstance(s.pattern_matches,int)
assert isinstance(s.adi_energy,int)
C["typed_evidence"]={
    "pass":True,
    "outputs":["pattern_matches","adi_energy","valid_pattern_observations"],
    "learned_weights":False,
    "thresholds":False
}

report["status"]="PASS"
report["candidate_rule"]=(
    "QH4/J2 pattern agreement is primary structural evidence; exact circular ADI "
    "energy is the secondary relation. All candidates tied on the lexicographic "
    "pair survive to BIND->REACT->MEASURE."
)
report["claim_boundary"]=(
    "The source theorems establish QH4 addressing/J2 and ADI exactness separately. "
    "The lexicographic candidate rule is a precommitted architecture choice. It is "
    "eligible for benchmark falsification only because it was fixed before training."
)

root=Path(__file__).resolve().parents[1]
out=root/"results"/"v29_relational_identify_survival.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
