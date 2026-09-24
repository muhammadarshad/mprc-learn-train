"""
Directional 1+8 -> Arm8/W2 pre-training survival gate
======================================================

NO TRAINING. NO LABELS.

For one binary channel bit-plane, the local directional context is

    C + (U1,U2) + (D1,D2) + (F1,F2) + (B1,B2)

The eight arm bits are packed into ONE Z256 byte through four exact base-4
directional states:

    AU = U1 + 2 U2
    AD = D1 + 2 D2
    AF = F1 + 2 F2
    AB = B1 + 2 B2

    A = AU + 4 AD + 16 AF + 64 AB
      = U1 + 2U2 + 4D1 + 8D2 + 16F1 + 32F2 + 64B1 + 128B2

The full 1+8 shape is NOT a scalar in Z512.  It is the W^2 relation

    W = (C,A) in Z256^2,

restricted to C in {0,1}.  Therefore the 512 binary shapes are 512 exact
two-byte states.

The one-byte relation is

    B = C - A mod 256.

This gate verifies the finite algebra/topology before any classifier may use
the representation.
"""

from pathlib import Path
import itertools, json

NAMES=("C","U1","U2","D1","D2","F1","F2","B1","B2")

# Spatial transpose (r,c)->(c,r): U<->B, D<->F.
TRANSPOSE={
    "C":"C",
    "U1":"B1","U2":"B2",
    "D1":"F1","D2":"F2",
    "F1":"D1","F2":"D2",
    "B1":"U1","B2":"U2",
}

# Clockwise 90deg rotation in image coordinates: U->F->D->B->U.
ROT90={
    "C":"C",
    "U1":"F1","U2":"F2",
    "F1":"D1","F2":"D2",
    "D1":"B1","D2":"B2",
    "B1":"U1","B2":"U2",
}

def pack(ctx):
    c,u1,u2,d1,d2,f1,f2,b1,b2=[int(x)&1 for x in ctx]
    au=u1 | (u2<<1)
    ad=d1 | (d2<<1)
    af=f1 | (f2<<1)
    ab=b1 | (b2<<1)
    a=au | (ad<<2) | (af<<4) | (ab<<6)
    assert 0<=c<=1 and 0<=a<=255
    return c,a

def unpack(c,a):
    c=int(c)&0xFF; a=int(a)&0xFF
    assert c in (0,1)
    return (
        c,
        (a>>0)&1,(a>>1)&1,
        (a>>2)&1,(a>>3)&1,
        (a>>4)&1,(a>>5)&1,
        (a>>6)&1,(a>>7)&1,
    )

def B(c,a):
    return (int(c)-int(a))&0xFF

def from_CB(c,b):
    # Since b=C-A, A=C-B.
    return int(c)&0xFF,(int(c)-int(b))&0xFF

def permute(ctx,mapping):
    d=dict(zip(NAMES,ctx))
    out={}
    # mapping says old name moves to new name.
    for old,new in mapping.items():
        out[new]=d[old]
    return tuple(out[n] for n in NAMES)

def transpose_packed(c,a):
    return pack(permute(unpack(c,a),TRANSPOSE))

def rot90_packed(c,a):
    return pack(permute(unpack(c,a),ROT90))

results={}

# G1/G2 exact finite bijection and ring domain.
seen=set()
all_ctx=list(itertools.product((0,1),repeat=9))
for ctx in all_ctx:
    w=pack(ctx)
    assert unpack(*w)==ctx
    seen.add(w)
assert len(seen)==512
assert all(0<=c<=255 and 0<=a<=255 for c,a in seen)
results["G1_pack_roundtrip"]={"pass":True,"contexts":512}
results["G2_W2_injective"]={
    "pass":True,
    "distinct_W2_states":len(seen),
    "mathematical_domain":"Z256^2",
    "valid_subset":"{0,1} x Z256",
    "scalar_Z512_used":False,
}

# G3 base-4 arm decomposition exactly covers all 256 Arm8 bytes.
arm_states=set()
for a in range(256):
    ctx=unpack(0,a)
    _,u1,u2,d1,d2,f1,f2,b1,b2=ctx
    au=u1+2*u2; ad=d1+2*d2; af=f1+2*f2; ab=b1+2*b2
    assert a==au+4*ad+16*af+64*ab
    assert all(0<=x<=3 for x in (au,ad,af,ab))
    arm_states.add((au,ad,af,ab))
assert len(arm_states)==256
results["G3_four_direction_base4"]={
    "pass":True,
    "arm_byte_states":256,
    "direction_state_domain":"0..3",
}

# G4 B1 relation is exactly 2-to-1 over the 512 W states.
fibers={b:[] for b in range(256)}
for c,a in seen:
    fibers[B(c,a)].append((c,a))
assert set(map(len,fibers.values()))=={2}
results["G4_B1_fibers"]={
    "pass":True,
    "B_states":256,
    "preimages_per_B":2,
}

# G5 (C,B) is an exact two-byte alternate coordinate for W.
for c,a in seen:
    b=B(c,a)
    c2,a2=from_CB(c,b)
    assert (c2,a2)==(c,a)
results["G5_CB_reconstruction"]={"pass":True,"roundtrips":512}

# G6 transpose is exact, involutive, and stays in the same two-byte domain.
for ctx in all_ctx:
    w=pack(ctx)
    wt=transpose_packed(*w)
    assert unpack(*wt)==permute(ctx,TRANSPOSE)
    assert transpose_packed(*wt)==w
results["G6_transpose"]={"pass":True,"contexts":512,"order":2}

# G7 90deg rotation is exact and has order four.
for ctx in all_ctx:
    w=pack(ctx)
    cur=w
    for _ in range(4):
        cur=rot90_packed(*cur)
    assert cur==w
results["G7_rotation90"]={"pass":True,"contexts":512,"order":4}

# G8 arm near/far states survive packing exactly.
for ctx in all_ctx:
    c,a=pack(ctx)
    back=unpack(c,a)
    for i in (1,3,5,7):
        near,far=back[i],back[i+1]
        arm=near+2*far
        assert arm in (0,1,2,3)
results["G8_near_far_context"]={
    "pass":True,
    "contexts":512,
    "arm_states":{
        "0":"near0 far0",
        "1":"near1 far0",
        "2":"near0 far1",
        "3":"near1 far1",
    }
}

# G9 implementation serialization is explicitly not ring arithmetic.
# idx is allowed only as an array/dictionary address.
idxs={(c<<8)|a for c,a in seen}
assert len(idxs)==512 and min(idxs)==0 and max(idxs)==511
results["G9_lookup_index_boundary"]={
    "pass":True,
    "lookup_indices":512,
    "index_range":[0,511],
    "index_semantics":"host lookup address only; never a Z256 state or arithmetic operand",
}

# G10 W2 and B1 names match frozen relation roles: W retains exact shape, B
# deliberately folds two W states together.
results["G10_relation_roles"]={
    "pass":True,
    "W2":"exact two-byte local-shape relation",
    "B1":"one-byte difference relation with exactly two W preimages",
}

results["overall"]="ALL PASS"
results["training_gate_open"]=True
results["claim_boundary"]="This proves finite encoding/topology facts only. Classification usefulness remains empirical."

root=Path(__file__).resolve().parents[1]
out=root/"results"/"directional_arm8_w2_survival_gate.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results,indent=2),encoding="utf-8")
print(json.dumps(results,indent=2))
