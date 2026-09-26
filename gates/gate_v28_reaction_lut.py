"""v28 — ReactionLUT ring-medoid survival gate.

NO IMAGE DATASET. NO LABELS. NO CLASSIFICATION.

The purpose is to decide whether one concrete observation-populated ReactionLUT
rule is mathematically and computationally coherent before it is allowed into
training.
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from mprc_structural.reaction_lut import CDIST, fit_from_counts

rng=np.random.default_rng(20260927)
report={"gate":"v28-ring-medoid-reaction-lut","checks":{}}
C=report["checks"]

for a in range(256):
    for b in range(256):
        d=int(CDIST[a,b])
        assert d==min((a-b)&255,(b-a)&255)
        assert d==int(CDIST[b,a])
        assert 0<=d<=128
C["cdist_table"]={"pass":True,"ordered_pairs":65536}

z=np.zeros((256,256),dtype=np.int64)
L=fit_from_counts(z)
assert np.array_equal(L,np.arange(256,dtype=np.uint8))
C["unseen_identity"]={"pass":True,"entries":256}

singleton_cases=0
for u in range(256):
    h=np.zeros((256,256),dtype=np.int64)
    for c in (0,64,128,192,u):
        h.fill(0)
        h[u,c]=1
        L=fit_from_counts(h)
        assert int(L[u])==c
        singleton_cases+=1
C["singleton_exact"]={"pass":True,"cases":singleton_cases}

counts=rng.integers(0,8,size=(256,256),dtype=np.int64)
L=fit_from_counts(counts)
optimal=0
for u in range(256):
    row=counts[u]
    cost=CDIST.astype(np.int64)@row
    assert int(cost[int(L[u])])==int(cost.min())
    optimal+=1
C["finite_argmin"]={"pass":True,"lut_entries":optimal,"candidates_per_entry":256}

base=rng.integers(0,5,size=(256,256),dtype=np.int64)
Lb=fit_from_counts(base)
shift_cases=0
for t in (0,1,7,31,64,127,128,183,255):
    shifted=np.zeros_like(base)
    for u in range(256):
        uu=(u+t)&255
        shifted[uu]=np.roll(base[u],t)
    Lt=fit_from_counts(shifted)
    expected=np.empty(256,dtype=np.uint8)
    for u in range(256):
        expected[(u+t)&255]=(int(Lb[u])+t)&255
    assert np.array_equal(Lt,expected)
    shift_cases+=256
C["ring_translation_equivariance"]={"pass":True,"entry_cases":shift_cases}

better=equal=0
for u in range(256):
    row=counts[u]
    learned=int((CDIST[int(L[u])].astype(np.int64)*row).sum())
    neutral=int((CDIST[u].astype(np.int64)*row).sum())
    assert learned<=neutral
    if learned<neutral:
        better+=1
    else:
        equal+=1
C["dominates_identity_on_objective"]={
    "pass":True,"strictly_better_entries":better,"equal_entries":equal
}

assert L.shape==(256,) and L.dtype==np.uint8
C["interface"]={"pass":True,"inputs":256,"outputs":256,"storage_bytes":256}

report["status"]="PASS"
report["theorem_status"]=(
    "Given the stated local reconstruction objective and observed counts, each LUT "
    "entry is an exact finite circular-L1 medoid. The optimizer is globally optimal "
    "for that objective because the 256 entries decouple."
)
report["candidate_status"]=(
    "The objective itself is a pre-training architecture choice, not a previously "
    "proved MPRC theorem. It is now eligible for a controlled training experiment "
    "because it was fixed and survived before seeing benchmark labels/results."
)
report["remaining_blocker"]=(
    "IDENTIFY still needs a frozen rule for how QH4 + corrected directional ADI "
    "evidence influences candidate/query state entering the attention/readout path."
)

root=Path(__file__).resolve().parents[1]
out=root/"results"/"v28_reaction_lut_survival.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
