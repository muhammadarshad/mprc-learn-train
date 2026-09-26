"""Exact Paper-3 QH4 mixed-radix / Transpose gate. No training."""

import json
from pathlib import Path
from mprc_structural.qh4_mixed import (
    VACUUM,mixed,active_address,decode_address,
    transpose_local,sum_coord,diff_coord
)

R={"gate":"Paper3 mixed-radix QH4 + local Transpose","checks":{}}
C=R["checks"]

seen=set()
quadrants={}
for th in range(4):
    vals=[]
    for a in range(3):
        for p in range(3):
            for s in range(1,8):
                m=mixed(th,a,p,s)
                vals.append(m)
                z=active_address(th,a,p,s)
                assert decode_address(z)==(th,a,p,s)
                seen.add(z)
    assert vals==list(range(64*th+1,64*th+64))
    quadrants[str(th)]=[min(vals),max(vals)]
assert len(seen)==252
assert set(range(256))-seen==set(VACUUM)
C["mixed_radix"]={"pass":True,"active":252,"quadrants":quadrants,"vacuum":sorted(VACUUM)}

orbit={}
for k in range(3):
    fixed=0; twocycles=0; visited=set()
    sectors={0:0,1:0,2:0}
    for th in range(4):
        for s in range(1,8):
            for a in range(3):
                for p in range(3):
                    sectors[sum_coord(a,p)]+=1
                    key=(th,a,p,s)
                    if key in visited: continue
                    a2,p2=transpose_local(a,p,k)
                    key2=(th,a2,p2,s)
                    assert transpose_local(a2,p2,k)==(a,p)
                    assert sum_coord(a2,p2)==sum_coord(a,p)
                    assert diff_coord(a2,p2)==(-(diff_coord(a,p)+2*k))%3
                    if key2==key:
                        fixed+=1; visited.add(key)
                    else:
                        twocycles+=1; visited.add(key); visited.add(key2)
    assert fixed==84 and twocycles==84 and len(visited)==252
    assert sectors=={0:84,1:84,2:84}
    orbit[str(k)]={"fixed":fixed,"two_cycles":twocycles,"invariant_sector_counts":sectors}
C["transpose"]={"pass":True,"k":orbit}
R["status"]="PASS"
R["claim_boundary"]=(
    "This closes the exact address-level 4 x 9 x 7 decomposition and Paper-3 local "
    "Transpose action. It does not define the semantic bijection from the corrected "
    "directional arm order (C,U1,U2,D1,D2,F1,F2,B1,B2) to the 3x3 (a,p) slots, "
    "nor the schedule selecting k during a ViT forward."
)
root=Path(__file__).resolve().parents[1]
out=root/"results"/"paper3_qh4_transpose_gate.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
