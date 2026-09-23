"""
Directional ADI-9 pre-training survival gate
=============================================

NO TRAINING. NO LABELS. NO CIFAR CLASSIFICATION.

The local MPRC context is

    a = (C,U1,U2,D1,D2,F1,F2,B1,B2) in Z_256^9

where the first depth is the frozen 5-arm context C,U1,D1,F1,B1 and
the second depth contributes U2,D2,F2,B2.

Forward directional ADI-9:

    Lambda = C+U1+U2+D1+D2+F1+F2+B1+B2              (mod 256)
    delta_j = C-a_j                                  (mod 256)

Because 9 is a unit of Z_256 and 9^{-1}=57,

    C = 57 * (Lambda + sum_j delta_j)                (mod 256)
    a_j = C-delta_j                                  (mod 256)

Hence the transform is an exact Z_256 automorphism.  Its matrix
determinant is 9, an odd unit.

The gate proves/checks:
 G1 geometry cardinality/in-bounds on 24,32,112,220,320
 G2 9*57 = 1 mod 256
 G3 forward/inverse matrices multiply to I in both orders
 G4 exhaustive scaled-basis round trip (9*256 states)
 G5 exhaustive binary-context round trip (all 2^9 = 512 contexts)
 G6 binary descriptor fingerprints are injective (512/512)
 G7 all ADI outputs remain bytes; no 512-state scalar is introduced
 G8 Lambda = S5(first depth) + outer4(second depth)
 G9 directional continuation follows from deltas:
      U1-U2 = delta_U2-delta_U1, etc.
 G10 transpose maps U<->B and D<->F at both depths exactly

Only after ALL PASS may this descriptor enter a training script.
"""

from pathlib import Path
import itertools, json, random

MOD=256
INV9=57
RESOLUTIONS=(24,32,112,220,320)

NAMES=("C","U1","U2","D1","D2","F1","F2","B1","B2")
OFFSETS={
    "C":(0,0),
    "U1":(-1,0),"U2":(-2,0),
    "D1":(1,0),"D2":(2,0),
    "F1":(0,1),"F2":(0,2),
    "B1":(0,-1),"B2":(0,-2),
}

# Under image transpose (r,c)->(c,r), using F=+column and B=-column.
TRANSPOSE_NAME={
    "C":"C",
    "U1":"B1","U2":"B2",
    "D1":"F1","D2":"F2",
    "F1":"D1","F2":"D2",
    "B1":"U1","B2":"U2",
}

def add(*xs):
    return sum(xs)&0xFF

def sub(a,b):
    return (a-b)&0xFF

def mul(a,b):
    return (a*b)&0xFF

def encode(a):
    assert len(a)==9
    a=[int(x)&0xFF for x in a]
    c=a[0]
    lam=sum(a)&0xFF
    ds=[sub(c,x) for x in a[1:]]
    out=[lam]+ds
    assert all(0<=x<=255 for x in out)
    return out

def decode(q):
    assert len(q)==9
    q=[int(x)&0xFF for x in q]
    lam=q[0]; ds=q[1:]
    c=mul(INV9,(lam+sum(ds))&0xFF)
    a=[c]+[sub(c,d) for d in ds]
    assert all(0<=x<=255 for x in a)
    return a

def matmul(A,B):
    m=len(A); p=len(B); n=len(B[0])
    assert len(A[0])==p
    return [[sum(A[i][k]*B[k][j] for k in range(p))&0xFF
             for j in range(n)] for i in range(m)]

def eye(n):
    return [[1 if i==j else 0 for j in range(n)] for i in range(n)]

# Forward matrix: Lambda row all +1; delta rows C-a_j.
M=[[1]*9]
for j in range(1,9):
    row=[0]*9
    row[0]=1
    row[j]=255  # -1 mod256
    M.append(row)

# Inverse matrix read directly from C=57*(Lambda+sum delta), a_j=C-delta_j.
MINV=[]
crow=[INV9]*9
MINV.append(crow)
for j in range(1,9):
    row=crow[:]
    row[j]=(row[j]-1)&0xFF
    MINV.append(row)

results={}

# G1: every valid center has exactly nine distinct in-bounds coordinates.
g1={}
for n in RESOLUTIONS:
    centers=0
    for r in range(2,n-2):
        for c in range(2,n-2):
            pts=[(r+OFFSETS[k][0],c+OFFSETS[k][1]) for k in NAMES]
            assert len(set(pts))==9
            assert all(0<=rr<n and 0<=cc<n for rr,cc in pts)
            centers+=1
    g1[str(n)]={"valid_centers":centers,"distinct_positions":9}
results["G1_geometry"]={"pass":True,"resolutions":g1}

# G2: odd-unit inverse.
assert (9*INV9)&0xFF==1
results["G2_inverse9"]={"pass":True,"inverse":INV9,"check":(9*INV9)&0xFF}

# G3: universal linear inverse certificate.
AB=matmul(M,MINV)
BA=matmul(MINV,M)
assert AB==eye(9) and BA==eye(9)
results["G3_matrix_inverse"]={"pass":True,"M_MINV_is_I":True,"MINV_M_is_I":True,
                              "determinant_integer":9,"determinant_is_unit":True}

# G4: exhaustive scaled basis. Since transform is Z256-linear and the
# matrices are exact inverses, this is a code-path regression over every
# scalar on every module generator.
count=0
for basis in range(9):
    for v in range(256):
        a=[0]*9
        a[basis]=v
        assert decode(encode(a))==a
        count+=1
results["G4_scaled_basis"]={"pass":True,"roundtrips":count}

# G5/G6/G7: every binary local context. This is the exact finite space the
# previous v1 incorrectly packed into a scalar 0..511.
fingerprints=set()
max_output=0
for bits in itertools.product((0,1),repeat=9):
    a=list(bits)
    q=encode(a)
    back=decode(q)
    assert back==a
    fingerprints.add(tuple(q))
    max_output=max(max_output,max(q))
assert len(fingerprints)==512
assert max_output<=255
results["G5_binary_roundtrip"]={"pass":True,"contexts":512}
results["G6_binary_injective"]={"pass":True,"distinct_fingerprints":len(fingerprints)}
results["G7_ring_native"]={
    "pass":True,
    "descriptor_components":9,
    "component_domain":"Z256/u8",
    "max_observed_descriptor_byte":max_output,
    "forbidden_512_state_scalar_used":False
}

# G8/G9: algebra identities. Exhaustively test all binary contexts and
# a deterministic large byte regression.
def identity_checks(a):
    c,u1,u2,d1,d2,f1,f2,b1,b2=a
    q=encode(a)
    lam=q[0]
    du1,du2,dd1,dd2,df1,df2,db1,db2=q[1:]
    s5=add(c,u1,d1,f1,b1)
    outer4=add(u2,d2,f2,b2)
    assert lam==add(s5,outer4)
    assert sub(u1,u2)==sub(du2,du1)
    assert sub(d1,d2)==sub(dd2,dd1)
    assert sub(f1,f2)==sub(df2,df1)
    assert sub(b1,b2)==sub(db2,db1)

for bits in itertools.product((0,1),repeat=9):
    identity_checks(list(bits))

rng=random.Random(20260924)
random_trials=200000
for _ in range(random_trials):
    identity_checks([rng.randrange(256) for _ in range(9)])
results["G8_S5_plus_outer4"]={"pass":True,"binary_exhaustive":512,
                              "byte_regression_trials":random_trials}
results["G9_directional_continuation"]={"pass":True,"binary_exhaustive":512,
                                        "byte_regression_trials":random_trials}

# G10: coordinate-level transpose law, exhaustive over every valid center
# for the requested resolutions.
transpose_centers=0
for n in RESOLUTIONS:
    for r in range(2,n-2):
        for c in range(2,n-2):
            for name in NAMES:
                dr,dc=OFFSETS[name]
                src=(r+dr,c+dc)
                transposed=(src[1],src[0])

                mapped=TRANSPOSE_NAME[name]
                mdr,mdc=OFFSETS[mapped]
                expected=(c+mdr,r+mdc)
                assert transposed==expected,(n,r,c,name,transposed,expected)
            transpose_centers+=1
results["G10_transpose"]={"pass":True,"centers_checked":transpose_centers,
                           "mapping":TRANSPOSE_NAME}

results["overall"]="ALL PASS"
results["training_gate_open"]=True
results["note"]="This gate certifies the local directional ADI-9 transform/topology only. It does not certify a classifier, learned LUT, or accuracy claim."

root=Path(__file__).resolve().parents[1]
out=root/"results"/"directional_adi9_survival_gate.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results,indent=2),encoding="utf-8")

print(json.dumps(results,indent=2))
