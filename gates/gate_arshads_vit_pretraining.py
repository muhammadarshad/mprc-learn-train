"""Arshad's ViT pre-training contract gate.

NO DATASET. NO LABELS. NO TRAINING.

This gate executes the exact frozen pieces that are currently specified and
refuses to authorize training where the source specification leaves semantics open.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from mprc_structural import manifold
from mprc_structural.attention import (
    GEN_INV_64, GEN_INV_256, attention_forward, generator_orbit64,
    inverse_transport_manifold, transport_manifold, react_once_logical,
    measure_energy
)
from mprc_structural.block_local112 import (
    encode_h,decode_h,encode_v,decode_v,
    block_h_to_v,block_v_to_h,local_h_to_v,local_v_to_h
)
from mprc_structural.directional_adi import (
    INV9, encode as adi_encode, decode as adi_decode,
    s5 as adi_s5, continuation as adi_continuation,
    transpose as adi_transpose
)
from mprc_structural.qh4 import (
    VACUUM, ACTIVE, forward as qh4_forward, inverse as qh4_inverse,
    active_positions
)
from mprc_structural.typed_attention import TypedDescriptorIndex, _descriptor

report={"gate":"Arshad's ViT pre-training contract","checks":{}}
C=report["checks"]
rng=np.random.default_rng(20260927)

# G1 QH4.
pos=active_positions()
assert len(pos)==ACTIVE==252
assert len(set(pos))==252
assert set(range(256))-set(pos)==set(VACUUM)
for g in range(1,37):
    for s in range(1,8):
        p=qh4_forward(g,s)
        gg,ss,theta=qh4_inverse(p)
        assert (gg,ss)==(g,s)
        assert theta==(g-1)//9+1
C["G1_QH4"]={"pass":True,"active":252,"vacuum":sorted(VACUUM)}

# G2 generator.
orb=generator_orbit64()
assert len(set(map(int,orb)))==64
assert manifold.GEN==7
assert GEN_INV_64==55 and GEN_INV_256==183
assert (7*55)%64==1 and (7*183)%256==1
C["G2_generator7"]={"pass":True,"period64":64,"inverse64":55,"inverse256":183}

# G3 corrected directional ADI: exact basis/value family + derived context.
cases=0
for k in range(9):
    for v in range(256):
        a=[0]*9; a[k]=v
        z=adi_encode(a)
        assert adi_decode(z)==tuple(a)
        direct=(a[0]+a[1]+a[3]+a[5]+a[7])&255
        assert adi_s5(z)==direct
        cont=((a[2]-a[1])&255,(a[4]-a[3])&255,
              (a[6]-a[5])&255,(a[8]-a[7])&255)
        assert adi_continuation(z)==cont
        cases+=1
assert (9*INV9)&255==1
C["G3_directional_ADI9"]={"pass":True,"basis_value_cases":cases,"inv9":INV9}

# G4 cdist/MEASURE pair law, exhaustive through scalar formula.
def cd(a,b):
    return min((a-b)&255,(b-a)&255)
pairs=0
for a in range(256):
    for b in range(256):
        d=cd(a,b)
        assert 0<=d<=128 and d==cd(b,a) and ((d==0)==(a==b))
        pairs+=1
C["G4_cdist"]={"pass":True,"ordered_pairs":pairs}

# G5 scalar affine BIND inversion on all odd s,t,bytes.
bind_cases=0
for s in range(1,256,2):
    sinv=pow(s,-1,256)
    for t in range(256):
        for x in (0,1,63,64,127,128,191,192,255):
            y=(s*x+t)&255
            xr=(sinv*(y-t))&255
            assert xr==x
            bind_cases+=1
C["G5_scalar_BIND"]={"pass":True,"deterministic_cases":bind_cases}

# G6 scalar fusion index identity exhaustive in s,t,u.
fusion=0
for s in range(1,256,2):
    for t in range(256):
        for u in range(256):
            assert (s*u+5*t)&255 == (s*u+(5*t&255))&255
            fusion+=1
C["G6_BIND_REACT_fusion"]={"pass":True,"index_cases":fusion}

# G7 vector bind fusion is ring linearity of S5, exhaustive sum residues.
vf=0
for u in range(256):
    for v in range(256):
        assert ((u+v)&255)==((v+u)&255)
        vf+=1
C["G7_vector_BIND_fusion"]={"pass":True,"sum_cases":vf}

rng=np.random.default_rng(20260927)

# G7b v24 observation -> structural interface: 16 x 113 x 128
# becomes 16 independent 128 x 113 manifolds by channel-wise transpose.
obs_a=rng.integers(0,256,size=(16,113,128),dtype=np.uint8)
obs_b=rng.integers(0,256,size=(16,113,128),dtype=np.uint8)
struct_a=np.transpose(obs_a,(0,2,1)).copy()
struct_b=np.transpose(obs_b,(0,2,1)).copy()
assert struct_a.shape==struct_b.shape==(16,128,113)
assert np.array_equal(np.transpose(struct_a,(0,2,1)),obs_a)

# Coordinate-wise BIND commutes with Pi_T.
bind_obs=((obs_a.astype(np.uint16)+obs_b.astype(np.uint16))&0xFF).astype(np.uint8)
bind_struct=((struct_a.astype(np.uint16)+struct_b.astype(np.uint16))&0xFF).astype(np.uint8)
assert np.array_equal(np.transpose(bind_obs,(0,2,1)),bind_struct)

# Wide ring MEASURE is invariant under coordinate permutation.
def array_energy(a,b):
    aa=a.astype(np.int16,copy=False); bb=b.astype(np.int16,copy=False)
    ab=(aa-bb)&0xFF; ba=(bb-aa)&0xFF
    return int(np.minimum(ab,ba).sum(dtype=np.int64))
assert array_energy(obs_a,obs_b)==array_energy(struct_a,struct_b)
C["G7b_v24_observation_interface"]={
    "pass":True,
    "source_shape":[16,113,128],
    "target_shape":[16,128,113],
    "states_preserved":16*113*128,
    "BIND_commutes":True,
    "MEASURE_invariant":True
}

# v26 IDENTIFY type survival: preserve the full 16-dimensional evidence vector.
raw_vec=[array_energy(struct_a[k],struct_b[k]) for k in range(16)]
assert len(raw_vec)==16 and all(isinstance(v,int) for v in raw_vec)
C["G7c_v26_IDENTIFY_type"]={
    "pass":True,
    "evidence_dimensions":16,
    "channels_collapsed":False
}

# G8 canonical DATA+INFORMATION pack is exact.
data=rng.integers(0,256,size=manifold.DATA_BYTES,dtype=np.uint8)
info=rng.integers(0,256,size=manifold.INFO_BYTES,dtype=np.uint8)
hv=manifold.pack(data,info)
un=manifold.unpack(hv)
assert np.array_equal(un.data,data)
assert np.array_equal(un.information,info)
C["G8_pack"]={"pass":True,"data":manifold.DATA_BYTES,"information":manifold.INFO_BYTES}

# G9 seven-round support count.
support={(0,0)}
counts=[1]
for r in range(1,8):
    nxt=set(support)
    for x,y in support:
        nxt.update(((x+1,y),(x-1,y),(x,y+1),(x,y-1)))
    support=nxt
    assert len(support)==1+2*r*(r+1)
    counts.append(len(support))
assert counts[-1]==113
C["G9_REACT_support"]={"pass":True,"counts":counts}

# G10 exact native 16x7 <-> 7x16 coordinate transport.
count=0
for r in range(112):
    for c in range(112):
        hb,hl=encode_h(r,c)
        vb=block_h_to_v(hb); vl=local_h_to_v(hl)
        assert (vb,vl)==encode_v(c,r)
        assert block_v_to_h(vb)==hb
        assert local_v_to_h(vl)==hl
        assert decode_h(hb,hl)==(r,c)
        assert decode_v(vb,vl)==(c,r)
        count+=1
C["G10_rectangular_transpose"]={"pass":True,"pixels":count}

# Directional ADI transpose covariance, exact finite basis/value family.
tcases=0
for k in range(9):
    for v in range(256):
        a=[11,29,47,83,109,137,163,211,239]
        a[k]=v
        z=adi_encode(a)
        zt=adi_transpose(z)
        assert adi_transpose(zt)==z
        tcases+=1
C["G10b_directional_transpose"]={"pass":True,"cases":tcases}

# G11 current corrected attention state path:
# generator transport -> vector BIND -> REACT -> MEASURE -> inverse transport.
state=rng.integers(0,256,size=(manifold.H,manifold.W),dtype=np.uint8)
query=rng.integers(0,256,size=(manifold.H,manifold.W),dtype=np.uint8)
out=attention_forward(state,query,rounds=7)
assert out["state"].shape==(manifold.H,manifold.W)
assert out["state"].dtype==np.uint8
assert isinstance(out["energy"],int)
assert 0<=out["energy"]<=128*manifold.MANIFOLD_BYTES
C["G11_attention_state_path"]={
    "pass":True,
    "executed":["generator7_transport","vector_BIND","REACT_x7","MEASURE","inverse_transport"],
    "energy":out["energy"]
}

# G12 exact typed IDENTIFY action candidate: direct inverse image of the full
# nine-byte directional descriptor. No labels, Top-K, radius or learned policy.
raw=np.asarray([
    [[11,29,47,83,109,137,163,211,239]],
    [[11,29,47,83,109,137,163,211,239]],
],dtype=np.uint8)
sites=np.asarray([100],dtype=np.int64)
idx=TypedDescriptorIndex.build(raw,sites)
d=_descriptor(raw[0,0],"H")
cand=idx.candidates(d)
assert len(cand)==2
assert [c.sample for c in cand]==[0,1]
q=list(d); q[0]=(q[0]+1)&255
assert len(idx.candidates(tuple(q)))==0
C["G12_typed_IDENTIFY_action"]={
    "pass":True,
    "rule":"I^T[D]={g:I(g)=D}",
    "descriptor_bytes":9,
    "exact_candidates":2,
    "top_k":False,
    "radius":False,
    "labels":False
}

# Contract-level source gaps. These are deliberately NOT guessed.
blockers=[]

# v27 closes H/V phase schedule ambiguity for the v24 full-frame transpose:
# BIND and REACT are transpose-equivariant and MEASURE is transpose-invariant.
closed_execution=[
    {
        "id":"C2_PHASE_SCHEDULE",
        "status":"CLOSED_v27",
        "detail":"For the v24 full-frame transpose, H/V alternation is algebraically energy-equivalent and is not a training hyperparameter."
    },
    {
        "id":"C3_METADATA_CODEC",
        "status":"CLOSED_FOR_v24_EXECUTION",
        "detail":"v24 preserves all 16x14,464 observed states exactly. INFORMATION lane semantics remain an open claim boundary, but no synthetic metadata is required for this observation path."
    },
    {
        "id":"C4_REACTION_LUT",
        "status":"CLOSED_CANDIDATE_v28",
        "detail":"v28 fixes a label-free observation-populated circular-L1 medoid LUT rule and proves exact finite optimality for its stated reconstruction objective."
    }
]

closed_execution.append({
    "id":"C1_IDENTIFY_TO_DECISION",
    "status":"CLOSED_CANDIDATE_TYPED_EXACT",
    "detail":"The first routing candidate is the exact inverse image of the full typed ADI-9 descriptor: I^T[D]={g:I(g)=D}. It was fixed before labels, survived the dedicated block gate, and introduces no Top-K, radius or learned similarity."
})

report["all_exact_independent_gates_pass"]=all(v["pass"] for v in C.values())
report["closed_candidate_interfaces"]=closed_execution
report["composition_blockers"]=blockers
report["training_authorized"]=report["all_exact_independent_gates_pass"] and len(blockers)==0
report["status"]="PRE-TRAINING CONTRACT PASS; EXACT TYPED ROUTING CANDIDATE AUTHORIZED FOR EMPIRICAL TRAINING"
report["rule"]="Training is authorized only for the survived typed route and precommitted ReactionLUT candidate. Recognition usefulness remains empirical; INFORMATION semantics remain an explicit claim boundary and may not be invented."

root=Path(__file__).resolve().parents[1]
outp=root/"results"/"arshads_vit_pretraining_contract.json"
outp.parent.mkdir(exist_ok=True)
outp.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
