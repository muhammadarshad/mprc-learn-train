"""Exact rational/integer learning-rate and directional accumulator gate.

The first claim is intentionally narrow:

    eta = 1/Q

does not require floating-point storage. A residual accumulator can preserve every
sub-quantum update exactly relative to the integer numerator stream.

This does NOT eliminate gradient descent; it eliminates floating representation of
its rational step. Higher-dimensional directional evidence is retained in four
separate Z4 direction bins rather than collapsed to one scalar.
"""

from __future__ import annotations

from fractions import Fraction
import json
import random
from pathlib import Path


def trunc_div(a:int,q:int)->int:
    """Truncate toward zero, unlike Python // for negative values."""
    if a >= 0:
        return a//q
    return -((-a)//q)


class ResidualUpdate:
    def __init__(self,Q:int):
        self.Q=int(Q)
        self.residual=0
        self.position=0
        self.emitted=0

    def add(self,g:int)->int:
        self.residual += int(g)
        k=trunc_div(self.residual,self.Q)
        self.residual -= k*self.Q
        self.position -= k
        self.emitted += abs(k)
        assert abs(self.residual) < self.Q
        return k


class DirectionalResidual4:
    """Four independent sub-quantum accumulators: +U,+D,-U,-D."""

    def __init__(self,Q:int):
        self.Q=int(Q)
        self.residual=[0,0,0,0]
        self.emitted=[0,0,0,0]

    def add(self,sq:int,magnitude:int):
        d=int(sq)&3
        m=int(magnitude)
        if m < 0:
            raise ValueError("magnitude must be non-negative")
        self.residual[d] += m
        k=self.residual[d]//self.Q
        self.residual[d] -= k*self.Q
        self.emitted[d] += k
        return d,k


report={}
rng=random.Random(20260925)

# G1 eta=1/1000 exact cumulative accounting over random signed gradient streams.
for Q in (10,100,1000,10000):
    for trial in range(1000):
        u=ResidualUpdate(Q)
        grads=[rng.randrange(-5000,5001) for _ in range(rng.randrange(1,500))]
        for g in grads:
            u.add(g)

        total=sum(grads)
        # Exact identity: total numerator = emitted signed quanta*Q + residual.
        # position = - emitted_signed.
        emitted_signed=-u.position
        assert total == emitted_signed*Q + u.residual

        rational = Fraction(total,Q)
        # Stored integer position is the trunc-toward-zero whole part of cumulative rational motion.
        assert emitted_signed == trunc_div(total,Q)

report["G1_rational_accumulator"]={
    "pass":True,
    "denominators":[10,100,1000,10000],
    "random_trials_per_denominator":1000,
}

# G2 cancellation is retained before emission.
u=ResidualUpdate(1000)
for _ in range(999): u.add(1)
assert u.position==0 and u.residual==999
u.add(-999)
assert u.position==0 and u.residual==0
report["G2_subquantum_cancellation"]={"pass":True}

# G3 exactly 1000 unit changes emit one quantum.
u=ResidualUpdate(1000)
for _ in range(1000): u.add(1)
assert u.position==-1 and u.residual==0
report["G3_one_thousand_changes"]={
    "pass":True,
    "eta":"1/1000",
    "unit_changes":1000,
    "whole_quanta_emitted":1,
}

# G4 four directions remain separate.
d=DirectionalResidual4(1000)
for _ in range(1000): d.add(0,1)  # +U
for _ in range(2000): d.add(1,1)  # +D
for _ in range(3000): d.add(2,1)  # -U
for _ in range(4000): d.add(3,1)  # -D
assert d.emitted == [1,2,3,4]
assert d.residual == [0,0,0,0]
report["G4_directional_Z4"]={
    "pass":True,
    "directions":["+U","+D","-U","-D"],
    "emitted":d.emitted,
}

# G5 mixed residuals never contaminate another direction.
d=DirectionalResidual4(1000)
for _ in range(999): d.add(0,1)
for _ in range(999): d.add(1,1)
d.add(0,1)
assert d.emitted == [1,0,0,0]
assert d.residual == [0,999,0,0]
report["G5_direction_isolation"]={"pass":True}

report["overall"]="ALL PASS"
report["claim_boundary"]=(
    "This proves exact integer accounting for rational learning-rate denominators "
    "and independent directional accumulation. It does not prove that gradient "
    "descent itself is unnecessary or that a particular direction encoding is optimal."
)

root=Path(__file__).resolve().parents[1]
out=root/"results"/"integer_learning_rate_gate.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
