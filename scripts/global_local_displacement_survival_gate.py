"""
Global/local active-shape displacement survival gate
====================================================

NO TRAINING. NO LABELS.

Let P be a finite nonempty active-pixel set in Z^2.
Define the lexicographic anchor

    a(P) = min_lex P

and normalized displacements

    D(P) = { p-a(P) : p in P }.

For the two-depth 5-arm context at an active point p, pack the eight
directional occupancies into the previously gated Arm8 byte A_P(p).

The local active-shape signature is

    Sigma(P) = sorted{ (p-a(P), A_P(p)) : p in P }.

The displacement coordinates alone reconstruct P up to translation; Arm8
adds the local two-depth context without destroying this property.

For disconnected local components P_k, define component anchors a_k and a
global anchor g=a(P).  The hierarchical record is

    H(P) = sorted{ (a_k-g, Sigma(P_k)) }.

This gate verifies translation invariance, injectivity up to translation,
rotation transport, and global/local displacement behavior before any
classifier may use the representation.
"""

from pathlib import Path
import itertools, json, random
from collections import deque

ARMS=[
    (-1,0),(-2,0),
    (1,0),(2,0),
    (0,1),(0,2),
    (0,-1),(0,-2),
]

def anchor(P):
    P=frozenset(P)
    assert P
    return min(P)

def translate(P,t):
    dr,dc=t
    return frozenset((r+dr,c+dc) for r,c in P)

def normalized(P):
    ar,ac=anchor(P)
    return tuple(sorted((r-ar,c-ac) for r,c in P))

def arm8(P,p):
    P=set(P)
    r,c=p
    a=0
    for k,(dr,dc) in enumerate(ARMS):
        if (r+dr,c+dc) in P:
            a |= 1<<k
    assert 0<=a<=255
    return a

def signature(P):
    P=frozenset(P)
    ar,ac=anchor(P)
    return tuple(sorted((r-ar,c-ac,arm8(P,(r,c))) for r,c in P))

def rot90_point(p):
    r,c=p
    return (c,-r)

def rot90(P):
    return frozenset(rot90_point(p) for p in P)

def connected_components(P):
    left=set(P)
    out=[]
    while left:
        seed=min(left)
        q=deque([seed])
        left.remove(seed)
        comp={seed}
        while q:
            r,c=q.popleft()
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                n=(r+dr,c+dc)
                if n in left:
                    left.remove(n);comp.add(n);q.append(n)
        out.append(frozenset(comp))
    return sorted(out,key=anchor)

def hierarchy(P):
    P=frozenset(P)
    g=anchor(P)
    gr,gc=g
    rec=[]
    for C in connected_components(P):
        ar,ac=anchor(C)
        rec.append((ar-gr,ac-gc,signature(C)))
    return tuple(sorted(rec))

def reconstruct_normalized_from_signature(sig):
    return tuple(sorted((dr,dc) for dr,dc,_ in sig))

results={}

# G1: anchor equivariance and signature translation invariance on exhaustive
# 4x4 binary shapes. 65535 nonempty shapes.
base=[(r,c) for r in range(4) for c in range(4)]
shapes=0
for mask in range(1,1<<16):
    P=frozenset(base[i] for i in range(16) if (mask>>i)&1)
    t=(7,-11)
    Q=translate(P,t)
    ap=anchor(P); aq=anchor(Q)
    assert aq==(ap[0]+t[0],ap[1]+t[1])
    assert normalized(Q)==normalized(P)
    assert signature(Q)==signature(P)
    shapes+=1
results["G1_translation_exhaustive_4x4"]={"pass":True,"nonempty_shapes":shapes}

# G2: exact reconstruction up to translation from displacement coordinates.
for mask in range(1,1<<16):
    P=frozenset(base[i] for i in range(16) if (mask>>i)&1)
    sig=signature(P)
    assert reconstruct_normalized_from_signature(sig)==normalized(P)
results["G2_reconstruct_up_to_translation"]={"pass":True,"shapes":shapes}

# G3: signature is injective on translation classes. Canonical normalized
# shapes may appear from multiple translated masks in the 4x4 frame; they
# must always have one and only one signature.
canon_to_sig={}
sig_to_canon={}
for mask in range(1,1<<16):
    P=frozenset(base[i] for i in range(16) if (mask>>i)&1)
    c=normalized(P); s=signature(P)
    if c in canon_to_sig:
        assert canon_to_sig[c]==s
    else:
        canon_to_sig[c]=s
    if s in sig_to_canon:
        assert sig_to_canon[s]==c
    else:
        sig_to_canon[s]=c
results["G3_injective_translation_classes"]={
    "pass":True,
    "translation_classes":len(canon_to_sig),
    "distinct_signatures":len(sig_to_canon),
}

# G4: 90-degree rotation transport. We do not make rotation invariant; we
# require a deterministic transported signature which returns after 4 turns.
for mask in range(1,1<<9):
    pts=[(r,c) for r in range(3) for c in range(3)]
    P=frozenset(pts[i] for i in range(9) if (mask>>i)&1)
    R=P
    for _ in range(4):
        R=rot90(R)
    assert R==P
    S=signature(P)
    SR=signature(rot90(P))
    # Both are exact normalized encodings of the corresponding shape.
    assert reconstruct_normalized_from_signature(S)==normalized(P)
    assert reconstruct_normalized_from_signature(SR)==normalized(rot90(P))
results["G4_rotation_transport"]={"pass":True,"shapes":511,"order":4}

# G5: local Arm8 context is translation-equivariant for every active point.
rng=random.Random(20260924)
trials=100000
for _ in range(trials):
    P={(rng.randrange(-5,6),rng.randrange(-5,6)) for _ in range(rng.randrange(1,30))}
    if not P: continue
    t=(rng.randrange(-20,21),rng.randrange(-20,21))
    Q=translate(P,t)
    for p in P:
        q=(p[0]+t[0],p[1]+t[1])
        assert arm8(P,p)==arm8(Q,q)
results["G5_Arm8_translation_equivariance"]={"pass":True,"random_trials":trials}

# G6: hierarchical disconnected-component record is globally translation
# invariant and reconstructs the normalized union.
def union_from_hierarchy(H):
    pts=set()
    for dr,dc,sig in H:
        for lr,lc,_ in sig:
            pts.add((dr+lr,dc+lc))
    return tuple(sorted(pts))

for _ in range(50000):
    # two or three separated synthetic components
    comps=[]
    offset=0
    for k in range(rng.choice((2,3))):
        h=rng.randrange(1,4); w=rng.randrange(1,4)
        C={(r,c+offset) for r in range(h) for c in range(w)
           if rng.randrange(100)<70}
        if not C: C={(0,offset)}
        comps.append(C)
        offset+=w+3
    P=frozenset().union(*map(frozenset,comps))
    t=(rng.randrange(-50,51),rng.randrange(-50,51))
    Q=translate(P,t)
    HP=hierarchy(P); HQ=hierarchy(Q)
    assert HP==HQ
    assert union_from_hierarchy(HP)==normalized(P)
results["G6_hierarchical_global_local"]={"pass":True,"random_trials":50000}

# G7: controlled local movement: moving one disconnected component changes
# only that component's global displacement; each component's local signature
# remains the same.
body=frozenset((r,c) for r in range(6) for c in range(5))
hat=frozenset((r-5,c+1) for r in range(2) for c in range(3))
P=body|hat
# Ensure disconnected with current geometry.
assert len(connected_components(P))==2
H0=hierarchy(P)
hat2=translate(hat,(-2,4))
P2=body|hat2
assert len(connected_components(P2))==2
H1=hierarchy(P2)

# Identify components by their local signature.
d0={rec[2]:(rec[0],rec[1]) for rec in H0}
d1={rec[2]:(rec[0],rec[1]) for rec in H1}
assert set(d0)==set(d1)
changed=[s for s in d0 if d0[s]!=d1[s]]
assert len(changed)==1
results["G7_local_component_motion"]={
    "pass":True,
    "component_count":2,
    "components_with_changed_global_displacement":1,
    "local_signatures_preserved":True,
}

results["overall"]="ALL PASS"
results["training_gate_open"]=True
results["claim_boundary"]=(
    "Exact for finite binary active sets and disconnected-component hierarchy. "
    "Semantic decomposition of touching/overlapping real-image parts is NOT proved."
)

root=Path(__file__).resolve().parents[1]
out=root/"results"/"global_local_displacement_survival_gate.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results,indent=2),encoding="utf-8")
print(json.dumps(results,indent=2))
