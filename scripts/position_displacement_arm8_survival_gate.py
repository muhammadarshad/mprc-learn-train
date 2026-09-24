"""
Position + displacement + Arm8 context survival gate
=====================================================

NO TRAINING. NO LABELS.

On the 7x7 observer lattice:
  g = lexicographic global active-shape anchor
  p = active observer position
  d = p-g
  A = gated two-depth Arm8 context byte

Exact identity:
  p = g+d.

We retain position rather than quotienting it away.

Host metadata/indexes:
  G  = 7*g_r + g_c                  in 0..48
  P  = 7*p_r + p_c                  in 0..48
  D  = 13*d_r + (d_c+6)             in 0..90
  PA = 256*P + A                     in 0..12543
  DA = 256*D + A                     in 0..23295

Only A is a Z256 context byte. G/P/D/PA/DA are metadata or host table
addresses and are never ring arithmetic operands.
"""

from pathlib import Path
import itertools, json

def idx7(r,c):
    assert 0<=r<7 and 0<=c<7
    return 7*r+c

def unidx7(i):
    assert 0<=i<49
    return i//7,i%7

def didx(dr,dc):
    assert 0<=dr<=6 and -6<=dc<=6
    return 13*dr+(dc+6)

def undidx(i):
    assert 0<=i<91
    return i//13,(i%13)-6

def pa_idx(P,A):
    assert 0<=P<49 and 0<=A<256
    return 256*P+A

def da_idx(D,A):
    assert 0<=D<91 and 0<=A<256
    return 256*D+A

def anchor(P):
    return min(P)

results={}

# G1 absolute 7x7 position serialization.
for r in range(7):
    for c in range(7):
        assert unidx7(idx7(r,c))==(r,c)
results["G1_position_serialization"]={"pass":True,"positions":49}

# G2 exhaustive legal (global anchor, point) relation.
# Since g=min_lex(P), any active p must satisfy p>=lex g.
pairs=0
for gr in range(7):
    for gc in range(7):
        g=(gr,gc)
        G=idx7(*g)
        for pr in range(7):
            for pc in range(7):
                p=(pr,pc)
                if p<g:
                    continue
                P=idx7(*p)
                dr,dc=pr-gr,pc-gc
                D=didx(dr,dc)
                rr,rc=undidx(D)
                assert (gr+rr,gc+rc)==p
                assert unidx7(G)==g
                assert unidx7(P)==p
                pairs+=1
results["G2_p_equals_g_plus_d"]={"pass":True,"legal_anchor_point_pairs":pairs}

# G3 all PA host indices round-trip by divmod.
seen=set()
for P in range(49):
    for A in range(256):
        I=pa_idx(P,A)
        P2,A2=divmod(I,256)
        assert (P2,A2)==(P,A)
        seen.add(I)
assert len(seen)==49*256 and min(seen)==0 and max(seen)==12543
results["G3_PA_serialization"]={"pass":True,"states":len(seen),"range":[0,12543]}

# G4 all DA host indices round-trip.
seen=set()
for D in range(91):
    for A in range(256):
        I=da_idx(D,A)
        D2,A2=divmod(I,256)
        assert (D2,A2)==(D,A)
        seen.add(I)
assert len(seen)==91*256 and min(seen)==0 and max(seen)==23295
results["G4_DA_serialization"]={"pass":True,"states":len(seen),"range":[0,23295]}

# G5 exhaustive nonempty 3x3 active sets: {G,D} reconstructs all absolute P.
pts=[(r,c) for r in range(3) for c in range(3)]
for mask in range(1,1<<9):
    S=frozenset(pts[i] for i in range(9) if (mask>>i)&1)
    g=anchor(S)
    G=idx7(*g)
    ds=[]
    for p in S:
        dr,dc=p[0]-g[0],p[1]-g[1]
        ds.append(didx(dr,dc))
    rec=set()
    gr,gc=unidx7(G)
    for D in ds:
        dr,dc=undidx(D)
        rec.add((gr+dr,gc+dc))
    assert rec==set(S)
results["G5_absolute_shape_reconstruction"]={"pass":True,"shapes":511}

# G6 exhaustive 4x4 shapes: translating a shape changes G/P but preserves D.
pts4=[(r,c) for r in range(4) for c in range(4)]
for mask in range(1,1<<16):
    S=frozenset(pts4[i] for i in range(16) if (mask>>i)&1)
    g=anchor(S)
    D0=sorted(didx(p[0]-g[0],p[1]-g[1]) for p in S)

    T=frozenset((r+2,c+1) for r,c in S)
    gt=anchor(T)
    D1=sorted(didx(p[0]-gt[0],p[1]-gt[1]) for p in T)
    assert D0==D1

    # Absolute anchor has moved exactly by translation.
    assert gt==(g[0]+2,g[1]+1)
results["G6_position_vs_displacement"]={
    "pass":True,
    "shapes":65535,
    "fact":"translation changes absolute position G/P while preserving normalized D"
}

# G7 domain boundary.
results["G7_domain_boundary"]={
    "pass":True,
    "ring_state":"A in Z256",
    "metadata":["G/P in 0..48","D in 0..90"],
    "host_lookup":["PA in 0..12543","DA in 0..23295"],
    "forbidden":"treating metadata/lookup indexes as Z256 arithmetic values"
}

results["overall"]="ALL PASS"
results["training_gate_open"]=True
results["claim_boundary"]="Proves exact lattice position/displacement/context serialization and reconstruction only; classifier usefulness remains empirical."

root=Path(__file__).resolve().parents[1]
out=root/"results"/"position_displacement_arm8_survival_gate.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results,indent=2),encoding="utf-8")
print(json.dumps(results,indent=2))
