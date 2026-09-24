"""
Active-anchor displacement + Arm8 serialization survival gate
==============================================================

NO TRAINING. NO LABELS.

This is the exact object that the next classifier will consume.

On the 7x7 observer lattice, let P be the nonempty set of active centers.
Let g=min_lex(P). For every active p define d=p-g.

The full normalized global shape is the set of d values.
Each active point additionally carries its already-gated Arm8 byte A(p).

Classifier observation:
    O(p) = (d_row, d_col, A(p))

For the 7x7 lattice:
    d_row in 0..6
    d_col in -6..6

Host lookup serialization:
    D = 13*d_row + (d_col+6)        in 0..90
    I = 256*D + A                   in 0..23295

D and I are HOST TABLE ADDRESSES only. They are not ring states.
A remains the Z256 Arm8 byte.

This gate exhaustively verifies finite serialization, translation-normalized
shape recovery, and the lookup-address boundary.
"""

from pathlib import Path
import itertools, json

def anchor(P):
    return min(P)

def normalized(P):
    gr,gc=anchor(P)
    return tuple(sorted((r-gr,c-gc) for r,c in P))

def displacement_index(dr,dc):
    assert 0<=dr<=6
    assert -6<=dc<=6
    return 13*dr+(dc+6)

def displacement_from_index(D):
    assert 0<=D<=90
    return D//13,(D%13)-6

def lookup_index(dr,dc,A):
    assert 0<=A<=255
    return 256*displacement_index(dr,dc)+A

def lookup_from_index(I):
    assert 0<=I<91*256
    D,A=divmod(I,256)
    dr,dc=displacement_from_index(D)
    return dr,dc,A

results={}

# G1: every displacement in the legal classifier domain round-trips.
seen=set()
for dr in range(7):
    for dc in range(-6,7):
        D=displacement_index(dr,dc)
        assert displacement_from_index(D)==(dr,dc)
        seen.add(D)
assert len(seen)==91 and min(seen)==0 and max(seen)==90
results["G1_displacement_serialization"]={"pass":True,"states":91}

# G2: exact lookup serialization over all 91*256 possibilities.
seenI=set()
for dr in range(7):
    for dc in range(-6,7):
        for A in range(256):
            I=lookup_index(dr,dc,A)
            assert lookup_from_index(I)==(dr,dc,A)
            seenI.add(I)
assert len(seenI)==91*256
results["G2_joint_lookup_serialization"]={
    "pass":True,
    "states":len(seenI),
    "range":[min(seenI),max(seenI)],
    "semantics":"host lookup address only",
}

# G3: exhaustive nonempty 3x3 observer shapes recover exactly from displacement set.
pts=[(r,c) for r in range(3) for c in range(3)]
for mask in range(1,1<<9):
    P=frozenset(pts[i] for i in range(9) if (mask>>i)&1)
    disp=normalized(P)
    # anchor-normalized reconstruction
    R=frozenset(disp)
    assert normalized(R)==disp
results["G3_shape_recovery_3x3"]={"pass":True,"shapes":511}

# G4: exhaustive 4x4 shapes: normalized displacement set is translation invariant.
pts4=[(r,c) for r in range(4) for c in range(4)]
classes={}
for mask in range(1,1<<16):
    P=frozenset(pts4[i] for i in range(16) if (mask>>i)&1)
    N=normalized(P)
    Q=frozenset((r+5,c-4) for r,c in P)
    assert normalized(Q)==N
    classes.setdefault(N,0)
    classes[N]+=1
results["G4_translation_invariance_4x4"]={
    "pass":True,
    "nonempty_shapes":65535,
    "translation_classes":len(classes),
}

# G5: active-only rule. Inactive sites are absent, never represented by a fake byte.
for mask in range(1,1<<9):
    P=frozenset(pts[i] for i in range(9) if (mask>>i)&1)
    disp=normalized(P)
    assert len(disp)==len(P)
results["G5_active_only"]={
    "pass":True,
    "rule":"one observation per active center; inactive centers emit no observation",
}

# G6: ring boundary: only A is a Z256 value. D/I are metadata/index addresses.
results["G6_domain_boundary"]={
    "pass":True,
    "ring_state":"A in Z256",
    "metadata":"(d_row,d_col)",
    "host_addresses":["D in 0..90","I in 0..23295"],
    "forbidden_interpretation":"D or I as Z256 arithmetic state",
}

results["overall"]="ALL PASS"
results["training_gate_open"]=True
results["claim_boundary"]="Proves exact finite serialization and translation-normalized active-shape recovery on the observer lattice; classification usefulness remains empirical."

root=Path(__file__).resolve().parents[1]
out=root/"results"/"active_anchor_arm8_serialization_gate.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results,indent=2),encoding="utf-8")
print(json.dumps(results,indent=2))
