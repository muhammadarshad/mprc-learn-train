"""v28 — exact QH4 o directional-ADI IDENTIFY type gate.

NO DATASET. NO LABELS. NO TRAINING.
"""

from __future__ import annotations
import json
from pathlib import Path
import itertools
from mprc_structural.identify_qh4_adi import (
    VacuumTag,ActiveTag,qh4_tag,qh4_untag,
    identify_directional,recover_directional
)

R={"gate":"v28 typed IDENTIFY: QH4 o directional ADI9","checks":{}}
C=R["checks"]

# G1 QH4 tag/untag is a bijection on all 256 ring bytes.
active=vac=0
for v in range(256):
    t=qh4_tag(v)
    assert qh4_untag(t)==v
    if isinstance(t,VacuumTag): vac+=1
    elif isinstance(t,ActiveTag): active+=1
    else: raise AssertionError(type(t))
assert active==252 and vac==4
C["byte_tag_bijection"]={"pass":True,"active":active,"vacuum":vac}

# G2 exact directional basis/value family through full ADI->QH4->ADI inverse.
basis=0
for k in range(9):
    for v in range(256):
        a=[0]*9;a[k]=v
        adi,tags=identify_directional(a)
        assert recover_directional(adi,tags)==tuple(a)
        basis+=1
C["directional_basis_value_roundtrip"]={"pass":True,"cases":basis}

# G3 all 512 binary 1+8 contexts also survive the byte-native interface exactly.
binary=0
for a in itertools.product((0,1),repeat=9):
    adi,tags=identify_directional(a)
    assert recover_directional(adi,tags)==a
    binary+=1
C["binary_context_roundtrip"]={"pass":True,"cases":binary}

# G4 deterministic non-binary family covering all coordinates and many ring positions.
seed=(11,29,47,83,109,137,163,211,239)
family=0
for k in range(9):
    for v in range(0,256,7):
        a=list(seed);a[k]=v
        adi,tags=identify_directional(a)
        assert tuple(qh4_untag(t) for t in tags)==adi
        assert recover_directional(adi,tags)==tuple(a)
        family+=1
C["nonbinary_family"]={"pass":True,"cases":family}

R["status"]="PASS"
R["theorem_status"]=(
    "Directional ADI9 lands in Z256^9. QH4 active/vacuum tagging is a bijection "
    "of Z256 onto four vacuum tags plus 252 active mixed-radix tags. Therefore "
    "component-wise QH4 tagging of the nine ADI bytes is lossless and exact."
)
R["claim_boundary"]=(
    "This closes IDENTIFY's type composition only. It does not define how the nine "
    "tagged relations alter/select the manifold state or query entering BIND, and "
    "does not define a generator/QH4 routing action. Those remain separate."
)

root=Path(__file__).resolve().parents[1]
out=root/"results"/"v28_typed_identify_qh4_adi.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
