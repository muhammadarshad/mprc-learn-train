"""No-dataset survival gate for exact typed IDENTIFY -> ArshadBlock.

NO DATASET. NO LABELS. NO TRAINING. NO MODEL SELECTION.

This gate tests the candidate rule
    I^T[D] = {g : I(g)=D}
for the full directional ADI-9 descriptor, then proves the retrieved manifold
actually executes generator-7 TRANSPORT -> BIND -> REACT -> MEASURE.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from mprc_structural.manifold import H, W, DATA_W
from mprc_structural.directional_adi import encode as adi_encode
from mprc_structural.typed_attention import (
    HV, TypedDescriptorIndex, ArshadBlock, _descriptor,
)
from mprc_structural.attention import (
    transport_manifold, bind, react, measure_energy,
)

R={"gate":"Typed IDENTIFY transpose + instrumented ArshadBlock","checks":{}}
C=R["checks"]

N=8
A=4
sites=np.asarray([113*10+10,113*20+20,113*30+30,113*40+40],dtype=np.int64)

base=np.asarray([11,29,47,83,109,137,163,211,239],dtype=np.uint8)

def raw_spatial_transpose(a):
    # C,U1,U2,D1,D2,F1,F2,B1,B2 -> C,B1,B2,F1,F2,D1,D2,U1,U2
    q=[int(x)&255 for x in a]
    return np.asarray([q[0],q[7],q[8],q[5],q[6],q[3],q[4],q[1],q[2]],dtype=np.uint8)

raw=np.empty((N,A,9),dtype=np.uint8)
ph=np.full((N,A),"H",dtype="<U1")

# Anchor 0 is the repeated exact relation.  Alternate H/V representations
# of the SAME geometry to prove orientation canonicalization.
for n in range(N):
    if n%2==0:
        raw[n,0]=base
        ph[n,0]="H"
    else:
        raw[n,0]=raw_spatial_transpose(base)
        ph[n,0]="V"

# Other anchors are deterministic one-coordinate perturbations.
for n in range(N):
    for a in range(1,A):
        v=base.copy()
        v[(n+a)%9]=(int(v[(n+a)%9])+17*a+n+1)&255
        raw[n,a]=v
        ph[n,a]="H"

idx=TypedDescriptorIndex.build(raw,sites,ph)
rec=idx.reconstruct()
assert np.array_equal(rec,idx.canonical_descriptors)
C["G1_sparse_typed_transpose_roundtrip"]={
    "pass":True,
    "occurrences":N*A,
    "unique_buckets":len(idx.buckets),
    "dense_256_pow_9_table":False,
}

# Exact H and V query forms must retrieve the same N global occurrences.
dH=_descriptor(base,"H")
dV=_descriptor(raw_spatial_transpose(base),"V")
assert dH==dV
candH=idx.candidates(dH)
candV=idx.candidates(dV)
gH=[c.global_position for c in candH]
gV=[c.global_position for c in candV]
truth=[n*HV+int(sites[0]) for n in range(N)]
assert gH==truth and gV==truth
C["G2_orientation_canonical_exact_retrieval"]={
    "pass":True,
    "H_recall":[len(gH),N],
    "V_recall":[len(gV),N],
    "same_globals":True,
}

# Every ADI coordinate is decision-relevant: a one-byte descriptor perturbation
# must exclude every exact base occurrence.  This works in descriptor space and
# therefore does not conflate the nine coordinates.
coord_cases=0
for j in range(9):
    q=list(dH)
    q[j]=(q[j]+1)&255
    got={c.global_position for c in idx.candidates(tuple(q))}
    assert got.isdisjoint(truth)
    coord_cases+=1
C["G3_all_ADI_coordinates_participate"]={"pass":True,"coordinate_cases":coord_cases}

# Build full candidate manifolds.  Same exact local descriptor occurs in all
# samples, while global INFORMATION context differs.  No labels.
manifolds=np.empty((N,H,W),dtype=np.uint8)
for n in range(N):
    # Deterministic nonuniform witness so generator transport and REACT are active.
    rr=np.arange(H,dtype=np.uint16)[:,None]
    cc=np.arange(W,dtype=np.uint16)[None,:]
    manifolds[n]=((17*rr+29*cc+13*n+7*rr*cc)&255).astype(np.uint8)
    # Sample-specific INFORMATION context.
    manifolds[n,:,DATA_W:]=((31*n+np.arange(W-DATA_W,dtype=np.uint16)[None,:])&255).astype(np.uint8)

query=manifolds[0].copy()
block=ArshadBlock(lut=None,rounds=7)
outH=block.forward(base,query,manifolds,idx,phase="H")
outV=block.forward(raw_spatial_transpose(base),query,manifolds,idx,phase="V")

assert outH["candidate_count"]==N
assert outV["candidate_count"]==N
assert [r["global_position"] for r in outH["candidates"]] == [r["global_position"] for r in outV["candidates"]]
assert any(r["sample"]==0 for r in outH["candidates"])
C["G4_block_candidate_identity"]={
    "pass":True,
    "candidates":N,
    "self_included":True,
    "global_identity_preserved":True,
}

# Exact reference equality for every candidate: manually recompute all stages.
energies={}
for n in range(N):
    tc=transport_manifold(manifolds[n])
    tq=transport_manifold(query)
    b=bind(tc,tq)
    r=react(b,lut=None,rounds=7)
    e=measure_energy(r,tq)
    energies[n]=e

by_sample={}
for row in outH["candidates"]:
    by_sample.setdefault(row["sample"],row["energy"])
assert by_sample==energies
C["G5_forward_reference_bit_exact"]={"pass":True,"samples":N}

# Instrumentation must prove every stage executes.  On this witness, stage
# buffers must also be non-degenerate.
required=[
    "IDENTIFY_ADI9_QH4","TYPED_TRANSPOSE","GENERATOR7_TRANSPORT",
    "BIND","REACT_x7","MEASURE"
]
for n,t in outH["sample_traces"].items():
    tr=t["trace"]
    assert tr["executed"]==required
    assert tr["physical_sha256"]!=tr["transported_sha256"]
    assert tr["transported_sha256"]!=tr["bound_sha256"]
    assert tr["bound_sha256"]!=tr["reacted_sha256"]
C["G6_stage_participation"]={"pass":True,"required":required,"samples":N}

# MEASURE must retain global-context discrimination for the same local descriptor.
distinct=len(set(energies.values()))
assert distinct>1
C["G7_same_local_different_global_measure"]={
    "pass":True,
    "distinct_energies":distinct,
    "samples":N,
}

# No hidden relation expansion: exact descriptor gets exactly its inverse image.
assert outH["candidate_count"]==N
C["G8_no_policy_tuning"]={
    "pass":True,
    "policy":"exact full typed descriptor inverse image",
    "top_k":False,
    "radius":False,
    "labels":False,
    "softmax":False,
    "qkv":False,
}

R["all_pass"]=all(v["pass"] for v in C.values())
R["status"]="PASS" if R["all_pass"] else "FAIL"
R["candidate_rule_status"]=(
    "SURVIVED MATH+CODE. Exact typed descriptor transpose is authorized as the "
    "first frozen-before-label routing candidate. This is not a claim that exact "
    "matching is sufficient for recognition."
)
R["training_authorized_if_parent_contract_passes"]=bool(R["all_pass"])
R["claim_boundary"]=(
    "The inverse-image routing and forward participation are exact. Recognition "
    "usefulness, candidate sparsity on natural images, ReactionLUT usefulness and "
    "classification accuracy remain empirical."
)

root=Path(__file__).resolve().parents[1]
outp=root/"results"/"typed_arshad_block_survival.json"
outp.parent.mkdir(exist_ok=True)
outp.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
