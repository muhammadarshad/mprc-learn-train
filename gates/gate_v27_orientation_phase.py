"""v27 — v24 orientation-phase survival gate.

NO DATASET. NO LABELS. NO TRAINING.

v24 defines the exact channel-wise frame transpose
    Pi_T: 113x128 <-> 128x113
and proves compatibility with the native 7x16 <-> 16x7 rectangles.

This gate asks whether the row/column phase alternation can be frozen before
training rather than tuned on a benchmark.

For coordinate-wise BIND, isotropic five-site REACT with one common 256-byte
LUT, and circular MEASURE:

    T(BIND(A,Q)) = BIND(TA,TQ)
    T(REACT(A))  = REACT(TA)
    MEASURE(TA,TQ) = MEASURE(A,Q)

Therefore inserting T between consecutive REACT rounds, while transporting the
query with the same phase, cannot change the final scalar energy.  With six
between-round transposes across seven REACT rounds, the final state itself is
bit-identical to the no-transpose seven-round state.

This is an execution theorem for the v24 candidate interface.  It does not
claim v24 is the unique possible MPRC orientation interface.
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np

ROUNDS=7

def bind(a,q):
    a=np.asarray(a,dtype=np.uint8)
    q=np.asarray(q,dtype=np.uint8)
    assert a.shape==q.shape
    return ((a.astype(np.uint16)+q.astype(np.uint16))&255).astype(np.uint8)

def react_once(x,lut):
    x=np.asarray(x,dtype=np.uint8)
    table=np.asarray(lut,dtype=np.uint8)
    assert table.shape==(256,)
    out=x.copy()
    acc=(
        x[1:-1,1:-1].astype(np.uint16)
        +x[:-2,1:-1].astype(np.uint16)
        +x[2:,1:-1].astype(np.uint16)
        +x[1:-1,:-2].astype(np.uint16)
        +x[1:-1,2:].astype(np.uint16)
    )&255
    out[1:-1,1:-1]=table[acc.astype(np.uint8)]
    return out

def react(x,lut,rounds):
    y=np.asarray(x,dtype=np.uint8).copy()
    for _ in range(int(rounds)):
        y=react_once(y,lut)
    return y

def measure(a,b):
    aa=np.asarray(a,dtype=np.int16)
    bb=np.asarray(b,dtype=np.int16)
    assert aa.shape==bb.shape
    ab=(aa-bb)&255
    ba=(bb-aa)&255
    return int(np.minimum(ab,ba).sum(dtype=np.int64))

def T(x):
    return np.asarray(x,dtype=np.uint8).T.copy()

def alternating_forward(state,query,lut,rounds=ROUNDS):
    # BIND once in the current phase.  Between each consecutive REACT round,
    # transpose both state and query so the phase alternates H/V/H/...
    x=bind(state,query)
    q=np.asarray(query,dtype=np.uint8).copy()
    for r in range(rounds):
        x=react_once(x,lut)
        if r+1<rounds:
            x=T(x)
            q=T(q)
    return x,q,measure(x,q)

rng=np.random.default_rng(20260927)
report={"gate":"v27-v24-orientation-phase","checks":{}}
C=report["checks"]

# Several deterministic nonlinear LUTs plus identity.
luts=[
    np.arange(256,dtype=np.uint8),
    ((np.arange(256,dtype=np.uint16)*7+19)&255).astype(np.uint8),
    ((np.arange(256,dtype=np.uint16)**2+31)&255).astype(np.uint8),
]
perm=np.arange(256,dtype=np.uint8)
rng.shuffle(perm)
luts.append(perm.copy())

# 1. BIND transpose covariance.
bind_cases=0
for shape in ((113,128),(128,113),(7,16),(16,7)):
    for _ in range(16):
        a=rng.integers(0,256,size=shape,dtype=np.uint8)
        q=rng.integers(0,256,size=shape,dtype=np.uint8)
        assert np.array_equal(T(bind(a,q)),bind(T(a),T(q)))
        bind_cases+=1
C["bind_transpose_covariance"]={"pass":True,"cases":bind_cases}

# 2. REACT transpose covariance, including nonlinear LUTs.
react_cases=0
for shape in ((113,128),(128,113),(7,16),(16,7)):
    for lut in luts:
        for _ in range(8):
            x=rng.integers(0,256,size=shape,dtype=np.uint8)
            assert np.array_equal(T(react_once(x,lut)),react_once(T(x),lut))
            react_cases+=1
C["react_transpose_covariance"]={"pass":True,"cases":react_cases,"luts":len(luts)}

# 3. MEASURE transpose invariance.
measure_cases=0
for shape in ((113,128),(128,113),(7,16),(16,7)):
    for _ in range(32):
        a=rng.integers(0,256,size=shape,dtype=np.uint8)
        b=rng.integers(0,256,size=shape,dtype=np.uint8)
        assert measure(a,b)==measure(T(a),T(b))
        measure_cases+=1
C["measure_transpose_invariance"]={"pass":True,"cases":measure_cases}

# 4. Seven-round alternating schedule is bit-identical to no-transpose
# because there are six between-round transposes and REACT commutes with T.
round_cases=0
energies=[]
for lut in luts:
    for _ in range(10):
        state=rng.integers(0,256,size=(128,113),dtype=np.uint8)
        query=rng.integers(0,256,size=(128,113),dtype=np.uint8)

        ref=react(bind(state,query),lut,ROUNDS)
        eref=measure(ref,query)

        alt,qalt,ealt=alternating_forward(state,query,lut,ROUNDS)
        assert alt.shape==(128,113)
        assert qalt.shape==(128,113)
        assert np.array_equal(alt,ref)
        assert np.array_equal(qalt,query)
        assert ealt==eref
        energies.append(eref)
        round_cases+=1
C["seven_round_alternation"]={
    "pass":True,
    "cases":round_cases,
    "between_round_transposes":6,
    "final_state_bit_exact_to_no_transpose":True,
    "energy_bit_exact":True
}

# 5. Odd number of total transpose applications changes orientation only;
# scalar energy remains identical when query follows the same frame.
odd_cases=0
for lut in luts:
    for _ in range(10):
        state=rng.integers(0,256,size=(128,113),dtype=np.uint8)
        query=rng.integers(0,256,size=(128,113),dtype=np.uint8)
        x=react(bind(state,query),lut,ROUNDS)
        e=measure(x,query)
        xt=T(x); qt=T(query)
        assert xt.shape==(113,128)
        assert measure(xt,qt)==e
        odd_cases+=1
C["orientation_only_energy_invariance"]={"pass":True,"cases":odd_cases}

report["status"]="PASS"
report["theorem_status"]=(
    "For the v24 full-frame transpose interface, coordinate-wise BIND, the frozen "
    "isotropic S5+LUT REACT, and circular MEASURE are transpose equivariant/invariant. "
    "Alternating H/V orientation between seven REACT rounds is therefore fixed by algebra, "
    "not a training hyperparameter."
)
report["claim_boundary"]=(
    "This closes phase-schedule ambiguity only for the v24 candidate full-frame transpose. "
    "It does not prove that v24 is the unique possible orientation interface, and it does "
    "not close IDENTIFY->state/query wiring or ReactionLUT learning."
)

root=Path(__file__).resolve().parents[1]
out=root/"results"/"v27_orientation_phase_survival.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
