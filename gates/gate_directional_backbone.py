#!/usr/bin/env python3
"""Finite exactness gate for directional MPRC vision backbone primitives.

This verifies primitives and transport covariance only.
It deliberately DOES NOT claim that the current training forward composes
QH4/generator-7/transpose into inference; that seam remains BLOCKED.
"""
import json
from pathlib import Path

INV9=57
GEN=7
GEN_INV=183
VAC={0,64,128,192}
OFF=((0,0),(-1,0),(-2,0),(1,0),(2,0),(0,1),(0,2),(0,-1),(0,-2))

def adi(a):
    c=a[0]&255
    return [sum(a)&255]+[((c-a[i])&255) for i in range(1,9)]

def iadi(z):
    c=(INV9*((z[0]+sum(z[1:]))&255))&255
    return [c]+[((c-d)&255) for d in z[1:]]

def trans(block):
    return [list(r) for r in zip(*block)]

def desc(block,r,c):
    return adi([block[r+dr][c+dc] for dr,dc in OFF])

def td(z):
    # transpose maps new U<-old B, new D<-old F, new F<-old D, new B<-old U
    return [z[0],z[7],z[8],z[5],z[6],z[3],z[4],z[1],z[2]]

checks={}

# QH4 252/252.
seen={}
for gamma in range(1,37):
    theta=(gamma-1)//9+1
    for sigma in range(1,8):
        p=(7*(7*(gamma-1)+sigma+(gamma-1)//9))&255
        assert p not in VAC and p not in seen
        seen[p]=(gamma,sigma,theta)
        m=(p*GEN_INV)&255
        adj=m-((m-1)//64)
        g2=(adj-1)//7+1
        s2=(adj-1)%7+1
        t2=(g2-1)//9+1
        assert (g2,s2,t2)==(gamma,sigma,theta)
assert len(seen)==252
assert set(range(256))-set(seen)==VAC
checks["qh4"]=252

# generator-7 permutations.
assert (GEN*GEN_INV)&255==1
for start in range(64):
    w=[(start+7*t)%64 for t in range(64)]
    assert len(set(w))==64
checks["stride7_starts"]=64

# Rectangular transpose and directional covariance.
block=[[((31*r+73*c+19*r*c)&255) for c in range(7)] for r in range(16)]
assert trans(trans(block))==block
tb=trans(block)
ncenters=0
for r in range(2,14):
    for c in range(2,5):
        z=desc(block,r,c)
        assert desc(tb,c,r)==td(z)
        assert td(td(z))==z
        ncenters+=1
checks["transpose_bytes"]=112
checks["directional_covariant_centers"]=ncenters

# Exhaustive byte values over every directional coordinate under transpose.
cases=0
for k in range(9):
    for v in range(256):
        a=[13,29,47,83,109,137,163,211,239]
        a[k]=v
        p=[[0]*5 for _ in range(5)]
        for val,(dr,dc) in zip(a,OFF):
            p[2+dr][2+dc]=val
        z=desc(p,2,2)
        assert desc(trans(p),2,2)==td(z)
        assert iadi(z)==a
        cases+=1
checks["transpose_value_cases"]=cases

# Seven-round cross dependency support.
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
checks["react_support"]=counts

# cdist all pairs.
def cdist(a,b):
    return min((a-b)&255,(b-a)&255)
pairs=0
for a in range(256):
    for b in range(256):
        d=cdist(a,b)
        assert 0<=d<=128
        assert d==cdist(b,a)
        assert (d==0)==(a==b)
        pairs+=1
checks["cdist_pairs"]=pairs

report={
    "gate":"directional_backbone_primitives",
    "status":"PASS",
    "checks":checks,
    "classification":"EXHAUSTIVELY VERIFIED primitives",
    "training_authorization":"BLOCKED: actual ArshadBlock composition must still wire QH4, generator-7 navigation and rectangular transport into the inference state path.",
}
out=Path("results/directional_backbone_survival.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
