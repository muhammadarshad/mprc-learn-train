"""Exact no-label gate for coherent-displacement SELECT."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from mprc_structural.coherent_displacement_select import (
    coherent_displacement_scores,select_coherent_max
)

rng=np.random.default_rng(20260927)
R={"gate":"coherent displacement SELECT","checks":{}}
C=R["checks"]
N=11; CH=3; H=W=7
mem=rng.integers(0,256,size=(N,CH,H,W,9),dtype=np.uint8)

# G1 self identity gets complete support.
full=CH*H*W
for n in range(N):
    s,d=coherent_displacement_scores(mem[n],mem)
    assert int(s[n])==full
    assert tuple(map(int,d[n]))==(0,0)
C["G1_self_complete"]={"pass":True,"samples":N,"support":full}

# G2 a common translation is recovered with the exact overlapping support.
q=mem[0]
shift=np.zeros_like(q)
shift[:,1:,2:,:]=q[:,:-1,:-2,:]
m=np.stack([shift,mem[1]],axis=0)
s,d=coherent_displacement_scores(q,m)
expected=CH*(H-1)*(W-2)
assert int(s[0])==expected,(int(s[0]),expected)
assert tuple(map(int,d[0]))==(1,2),tuple(map(int,d[0]))
C["G2_common_displacement"]={"pass":True,"displacement":[1,2],"support":expected}

# G3 arbitrary occurrence permutation must NOT be a symmetry.
p=rng.permutation(H*W)
scr=q.reshape(CH,H*W,9)[:,p,:].reshape(CH,H,W,9)
s,d=coherent_displacement_scores(q,np.stack([scr],axis=0))
assert int(s[0])<full
C["G3_permutation_breaks_identity"]={"pass":True,"support_after_permutation":int(s[0]),"full":full}

# G4 changing one site reduces support by exactly one at zero displacement.
one=q.copy()
one[1,3,4,0]^=np.uint8(1)
s,d=coherent_displacement_scores(q,np.stack([one],axis=0))
assert int(s[0])==full-1
assert tuple(map(int,d[0]))==(0,0)
C["G4_one_site_contradiction"]={"pass":True,"support":int(s[0])}

# G5 ties survive; no arbitrary Top-K.
m2=np.stack([q.copy(),q.copy(),mem[2]],axis=0)
sel,s,d=select_coherent_max(q,m2)
assert set(map(int,sel))=={0,1}
C["G5_exact_ties_survive"]={"pass":True,"selected":list(map(int,sel))}

# G6 unrelated exact descriptors give no coherent support and abstain.
z=np.bitwise_xor(q,np.uint8(0xA5))
while int(coherent_displacement_scores(q,np.stack([z]))[0].max())!=0:
    z=(z+np.uint8(1)).astype(np.uint8)
sel,s,d=select_coherent_max(q,np.stack([z]))
assert len(sel)==0
C["G6_zero_support_abstains"]={"pass":True}

R["status"]="PASS";R["all_pass"]=all(v["pass"] for v in C.values())
R["rule"]="M_n(Q)=max_(dr,dc) count exact ADI occurrences agreeing under one common displacement."
R["properties"]={
 "translation_compatible":True,
 "arbitrary_position_permutation_invariant":False,
 "retains_displacement_P":True,
 "exact_descriptor_only":True,
 "top_k":False,"threshold":False,"softmax":False,"labels":False
}
root=Path(__file__).resolve().parents[1]
out=root/"results"/"coherent_displacement_select_survival.json"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
