"""No-data parity gate for sorted typed SELECT index."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from mprc_structural.typed_select import exact_match_counts,select_exact_max
from mprc_structural.typed_select_index import SortedTypedSelectIndex

rng=np.random.default_rng(20260927)
R={"gate":"sorted typed SELECT parity","checks":{}}
C=R["checks"]

cases=0
for N in (1,2,7,31,127):
    for F in (1,3,17,49):
        mem=rng.integers(0,256,size=(N,F,9),dtype=np.uint8)
        idx=SortedTypedSelectIndex(mem)

        queries=[mem[min(N-1,0)].copy()]
        for _ in range(8):
            queries.append(rng.integers(0,256,size=(F,9),dtype=np.uint8))
        if F>1:
            q=mem[0].copy()
            q[F//2,4]=(int(q[F//2,4])+1)&255
            queries.append(q)

        for q in queries:
            c0=exact_match_counts(q,mem)
            s0,_=select_exact_max(q,mem)
            c1=idx.match_counts(q)
            s1,_=idx.select(q)
            assert np.array_equal(c0,c1)
            assert np.array_equal(s0,s1)
            cases+=1

C["exact_reference_parity"]={"pass":True,"queries":cases}

# Explicit duplicates/ties.
mem=rng.integers(0,256,size=(20,11,9),dtype=np.uint8)
mem[19]=mem[3]
idx=SortedTypedSelectIndex(mem)
s0,c0=select_exact_max(mem[3],mem)
s1,c1=idx.select(mem[3])
assert np.array_equal(c0,c1)
assert np.array_equal(s0,s1)
assert set(map(int,s1))=={3,19}
C["duplicate_tie_parity"]={"pass":True,"selected":[3,19]}

R["status"]="PASS"; R["all_pass"]=True
R["training_authorization"]="Sorted index is exactly equivalent to the survived SELECT rule."
root=Path(__file__).resolve().parents[1]
out=root/"results"/"typed_select_index_survival.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
