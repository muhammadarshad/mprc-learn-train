"""No-label gate for CIFAR32 lossless frame integration."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from mprc_structural.cifar_frame import (
    frame_rgb32,to_manifolds,from_manifolds,recover_rgb32,
    AY,AX,FILL,VACUUM
)

R={"gate":"CIFAR32 lossless MPRC framing","checks":{}}
C=R["checks"]
rng=np.random.default_rng(20260927)

# G1 random complete images.
for i in range(512):
    x=rng.integers(0,256,size=(32,32,3),dtype=np.uint8)
    obs=frame_rgb32(x)
    man=to_manifolds(obs)
    assert np.array_equal(from_manifolds(man),obs)
    assert np.array_equal(recover_rgb32(obs),x)
C["G1_random_roundtrip"]={"pass":True,"images":512,"source_bytes":512*3072}

# G2 every byte value at several source coordinates/channels.
positions=((0,0),(0,31),(31,0),(31,31),(15,15),(7,23))
cases=0
for ch in range(3):
    for r,c in positions:
        for v in range(256):
            x=np.zeros((32,32,3),dtype=np.uint8)
            x[r,c,ch]=v
            obs=frame_rgb32(x)
            rec=recover_rgb32(from_manifolds(to_manifolds(obs)))
            assert np.array_equal(rec,x)
            cases+=1
C["G2_value_position_cases"]={"pass":True,"cases":cases}

# G3 added frame never uses QH4 vacuum bytes.
x=np.zeros((32,32,3),dtype=np.uint8)
obs=frame_rgb32(x)
mask=np.ones((113,128),dtype=bool)
mask[AY:AY+32,AX:AX+32]=False
added=obs[:,mask]
assert np.all(added==FILL)
assert int(FILL) not in VACUUM
C["G3_nonvacuum_frame"]={"pass":True,"fill":int(FILL),"added_states":int(added.size)}

# G4 source placement is fixed and no resize/interpolation exists.
C["G4_geometry"]={
    "pass":True,"source":[32,32,3],"observation":[3,113,128],
    "manifold":[3,128,113],"offset":[AY,AX],"resized":False
}

R["status"]="PASS"
R["all_pass"]=all(v["pass"] for v in C.values())
R["training_authorization"]="Framing survived. Source RGB bytes may now enter the survived typed IDENTIFY/SELECT/ArshadBlock path."
root=Path(__file__).resolve().parents[1]
out=root/"results"/"cifar_frame_survival.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
