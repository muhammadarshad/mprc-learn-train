"""Bit-exact batch attention gate. No dataset, labels or training."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from mprc_structural.attention import attention_forward
from mprc_structural.batch_attention import batch_energy
from mprc_structural.manifold import H,W

rng=np.random.default_rng(20260927)
R={"gate":"batch attention parity","checks":{}}
C=R["checks"]

cases=0
for channels in (1,3,7):
    for n in (1,2,5):
        candidates=rng.integers(0,256,size=(n,channels,H,W),dtype=np.uint8)
        query=rng.integers(0,256,size=(channels,H,W),dtype=np.uint8)
        for rounds in (1,2,7):
            for lut_kind in ("identity","random"):
                lut=None if lut_kind=="identity" else rng.integers(0,256,size=256,dtype=np.uint8)
                got=batch_energy(candidates,query,lut=lut,rounds=rounds)
                exp=[]
                for i in range(n):
                    e=0
                    for ch in range(channels):
                        out=attention_forward(
                            candidates[i,ch],query[ch],lut=lut,rounds=rounds,
                            return_physical_state=False,
                        )
                        e+=int(out["energy"])
                    exp.append(e)
                exp=np.asarray(exp,dtype=np.int64)
                assert np.array_equal(got,exp)
                cases+=n*channels

C["reference_parity"]={"pass":True,"candidate_channel_cases":cases}
R["status"]="PASS"
R["all_pass"]=True
R["training_authorization"]="Batch implementation is bit-exact with the survived scalar forward and may be used for training/evaluation."
root=Path(__file__).resolve().parents[1]
out=root/"results"/"batch_attention_survival.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
