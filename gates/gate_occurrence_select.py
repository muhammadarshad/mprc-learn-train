"""No-dataset survival gate for displacement-compatible occurrence SELECT."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from mprc_structural.directional_adi import transpose as adi_transpose
from mprc_structural.occurrence_select import (
    multiset_match_counts,select_multiset_max,SortedOccurrenceSelectIndex
)

rng=np.random.default_rng(20260927)
R={"gate":"global occurrence multiset SELECT","checks":{}}
C=R["checks"]

# Use unique-ish random descriptor fields to make mutation accounting exact.
N=13; A=31
mem=rng.integers(0,256,size=(N,A,9),dtype=np.uint8)
idx=SortedOccurrenceSelectIndex(mem)

# G1 optimized index equals simple mathematical reference.
cases=0
queries=[mem[i].copy() for i in range(N)]
queries += [rng.integers(0,256,size=(A,9),dtype=np.uint8) for _ in range(20)]
for q in queries:
    c0=multiset_match_counts(q,mem)
    c1=idx.match_counts(q)
    s0,_=select_multiset_max(q,mem)
    s1,_=idx.select(q)
    assert np.array_equal(c0,c1)
    assert np.array_equal(s0,s1)
    cases+=1
C["G1_index_reference_parity"]={"pass":True,"queries":cases}

# G2 self gets complete A-match support.
for n in range(N):
    c=idx.match_counts(mem[n])
    assert int(c[n])==A
    assert int(c.max())==A
C["G2_self_complete"]={"pass":True,"samples":N,"support":A}

# G3 arbitrary occurrence-position permutation is exactly invariant.
perm_cases=0
for n in range(N):
    base=idx.match_counts(mem[n])
    for _ in range(32):
        p=rng.permutation(A)
        got=idx.match_counts(mem[n,p,:])
        assert np.array_equal(base,got)
        perm_cases+=1
C["G3_displacement_permutation_invariance"]={"pass":True,"cases":perm_cases}

# G4 common H/V directional transpose keeps exact multiset relation.
def tf(x):
    y=np.empty_like(x)
    for i,z in enumerate(x):
        y[i]=np.asarray(adi_transpose(tuple(map(int,z))),dtype=np.uint8)
    return y
memT=np.empty_like(mem)
for n in range(N): memT[n]=tf(mem[n])
idxT=SortedOccurrenceSelectIndex(memT)
for n in range(N):
    assert np.array_equal(idx.match_counts(mem[n]),idxT.match_counts(tf(mem[n])))
C["G4_HV_canonical_invariance"]={"pass":True,"samples":N}

# G5 multiset duplicates are counted only up to min multiplicity.
d=mem[0,0].copy()
q=np.stack([d,d,d,mem[0,1]],axis=0).astype(np.uint8)
m=np.empty((2,5,9),dtype=np.uint8)
m[0]=np.stack([d,d,mem[0,2],mem[0,3],mem[0,4]])
m[1]=np.stack([d,d,d,d,mem[0,1]])
c=multiset_match_counts(q,m)
# sample0 gets min(3,2)=2; sample1 gets min(3,4)=3 plus mem[0,1]=1
assert c.tolist()==[2,4]
C["G5_multiset_multiplicity"]={"pass":True,"scores":c.tolist()}

# G6 exact ties survive.
m2=np.concatenate([mem,mem[[0]]],axis=0)
idx2=SortedOccurrenceSelectIndex(m2)
sel,c=idx2.select(mem[0])
assert set(map(int,sel))=={0,N}
C["G6_ties_survive"]={"pass":True,"selected":list(map(int,sel))}

# G7 no exact shape evidence => abstain.
q=np.bitwise_xor(mem[0],np.uint8(0xA5))
idx0=SortedOccurrenceSelectIndex(mem)
while int(idx0.match_counts(q).max())!=0:
    q=(q+np.uint8(1)).astype(np.uint8)
sel,c=idx0.select(q)
assert len(sel)==0 and int(c.max())==0
C["G7_zero_support_abstains"]={"pass":True}

R["status"]="PASS";R["all_pass"]=all(v["pass"] for v in C.values())
R["rule"]="M_n(Q)=sum_D min(c_Q(D),c_n(D)); keep all max-support samples; abstain at zero."
R["properties"]={
    "position_permutation_invariant":True,
    "exact_descriptor_only":True,
    "top_k":False,"threshold":False,"radius":False,"labels":False
}
R["training_authorization"]="Occurrence SELECT survived and may replace the fixed-slot control in the CIFAR training candidate."
root=Path(__file__).resolve().parents[1]
out=root/"results"/"occurrence_select_survival.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
