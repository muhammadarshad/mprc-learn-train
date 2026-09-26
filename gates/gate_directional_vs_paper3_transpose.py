"""Directional spatial transpose vs Paper-3 local Transpose compatibility gate.

NO TRAINING.

The corrected 1+8 directional context has spatial transpose:
  C fixed
  U1 <-> B1
  U2 <-> B2
  D1 <-> F1
  D2 <-> F2

Cycle type: 1^1 2^4.

Paper-3 T_k on Z3^2 has, for every k:
  3 fixed points + 3 two-cycles.

Cycle type: 1^3 2^3.

Conjugate permutations have identical cycle type. Therefore no bijection from the
nine directional slots to Z3^2 can make ordinary spatial transpose equal to any T_k.
"""

import json
from pathlib import Path
from mprc_structural.qh4_mixed import transpose_local

DIR=("C","U1","U2","D1","D2","F1","F2","B1","B2")
SPATIAL={
    "C":"C",
    "U1":"B1","B1":"U1",
    "U2":"B2","B2":"U2",
    "D1":"F1","F1":"D1",
    "D2":"F2","F2":"D2",
}

def cycles(items,fn):
    seen=set(); out=[]
    for x in items:
        if x in seen: continue
        cyc=[]; y=x
        while y not in seen:
            seen.add(y);cyc.append(y);y=fn(y)
        out.append(tuple(cyc))
    return sorted([len(c) for c in out])

sp=cycles(DIR,lambda x:SPATIAL[x])
assert sp==[1,2,2,2,2]

paper={}
pts=[(a,p) for a in range(3) for p in range(3)]
for k in range(3):
    ck=cycles(pts,lambda x:transpose_local(x[0],x[1],k))
    assert ck==[1,1,1,2,2,2]
    paper[str(k)]=ck
    assert ck!=sp

report={
    "gate":"Directional spatial transpose / Paper3 T_k compatibility",
    "status":"PASS: NON-EQUIVALENCE PROVED",
    "directional_spatial_cycle_type":sp,
    "paper3_Tk_cycle_types":paper,
    "theorem":(
        "No bijection f from the nine corrected directional slots to Z3^2 can satisfy "
        "f o SpatialTranspose o f^-1 = T_k for any k in Z3, because conjugate "
        "permutations preserve cycle type and the fixed-point counts are 1 versus 3."
    ),
    "consequence":(
        "Native image H<->V spatial transpose and Paper3 local Arshad T_k are distinct "
        "operators. Do not use T_k as the byte spatial-transpose rule and do not infer "
        "the directional-slot -> (a,p) mapping from transpose covariance."
    )
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"directional_vs_paper3_transpose_gate.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
