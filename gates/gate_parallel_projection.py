"""Gate: synchronous parallel projection semantics."""
import itertools
import numpy as np
from mprc_structural.parallel_projection import DIRECTIONS,parallel_resolve,cdist

def toward(cur,ctx):
    # one ring step toward context; ties at 128 preserve current
    cw=(ctx-cur)&255; ccw=(cur-ctx)&255
    if cw<ccw:return (cur+1)&255
    if ccw<cw:return (cur-1)&255
    return cur

x=np.asarray([5,40,90,140,200,250,20,70,130],dtype=np.uint8)
base=parallel_resolve(x,toward)

# G1: input snapshot never mutates.
assert np.array_equal(x,np.asarray([5,40,90,140,200,250,20,70,130],dtype=np.uint8))
assert np.array_equal(base["snapshot"],x)

# G2: execution schedule cannot affect ANY proposal or final vote.
for order in itertools.permutations(DIRECTIONS):
    z=parallel_resolve(x,toward,order)
    assert np.array_equal(z["state"],base["state"])
    assert np.array_equal(z["mutated"],base["mutated"])
    for d in DIRECTIONS:
        assert np.array_equal(z["proposals"][d],base["proposals"][d])

# G3: branches really differ, proving they are directional rather than aliases.
assert any(not np.array_equal(base["proposals"][DIRECTIONS[0]],base["proposals"][d]) for d in DIRECTIONS[1:])

# G4: this diagnostic must actually resolve at least one site; a no-op vote
# must never be allowed to produce a false-green synchrony gate.
assert bool(base["mutated"].any())

# G5: every accepted mutation has at least one branch proposing mutation.
for i,m in enumerate(base["mutated"]):
    if m:
        assert any(base["proposals"][d][i]!=x[i] for d in DIRECTIONS)

# G6: directional one-step resolver never increases local context distance.
for a in range(256):
    for b in range(256):
        z=toward(a,b)
        assert cdist(z,b)<=cdist(a,b)

print({
 "gate":"synchronous parallel directional projection",
 "status":"PASS",
 "directions":DIRECTIONS,
 "schedule_permutations_checked":24,
 "snapshot":x.tolist(),
 "resolved":base["state"].tolist(),
 "mutated_sites":np.flatnonzero(base["mutated"]).tolist(),
})
