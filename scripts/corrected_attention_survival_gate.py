"""Machine-readable survival gate for corrected MPRC attention topology."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mprc_structural.attention import (
    GEN_INV_64,
    GEN_INV_256,
    attention_forward,
    generator_orbit64,
    inverse_transport_manifold,
    logical_physical_row,
    react_once_logical,
    transport_manifold,
)
from mprc_structural.manifold import (
    D,
    N,
    GEN,
    H,
    W,
    DATA_W,
    INFO_W,
    DATA_BYTES,
    INFO_BYTES,
    MANIFOLD_BYTES,
    TILE_H,
    SLAB_BYTES,
)

report = {}

# G1 corrected manifold arithmetic.
assert D == 256
assert N == 15
assert GEN == 7
assert H == 128
assert W == 113
assert DATA_W == 98
assert INFO_W == 15
assert DATA_BYTES == 12_544
assert INFO_BYTES == 1_920
assert MANIFOLD_BYTES == 14_464
assert SLAB_BYTES == 7_232
report["G1_manifold"] = {
    "pass": True,
    "data_bytes": DATA_BYTES,
    "information_bytes": INFO_BYTES,
    "shape": [H, W],
    "execution_slab": [TILE_H, W],
}

# G2 generator-7 exact orbit/inverses.
orbit = generator_orbit64()
assert len(set(map(int, orbit))) == 64
assert (GEN * GEN_INV_64) % 64 == 1
assert (GEN * GEN_INV_256) % 256 == 1
report["G2_generator7"] = {
    "pass": True,
    "period": 64,
    "inverse_mod64": GEN_INV_64,
    "inverse_mod256": GEN_INV_256,
}

# G3 transport/inverse exact random regression.
rng = np.random.default_rng(20260925)
trials = 200
for _ in range(trials):
    x = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    assert np.array_equal(inverse_transport_manifold(transport_manifold(x)), x)
report["G3_transport_roundtrip"] = {"pass": True, "random_trials": trials}

# G4 every manifold coordinate visited exactly once across two cache slabs.
seen = set()
for s in range(2):
    for t in range(64):
        r = s * 64 + int(orbit[t])
        for w in range(W):
            key = (r, w)
            assert key not in seen
            seen.add(key)
assert len(seen) == MANIFOLD_BYTES
report["G4_full_state_coverage"] = {
    "pass": True,
    "visited": len(seen),
    "expected": MANIFOLD_BYTES,
}

# G5 generator is topology, not merely execution order.
logical_row = 31
col = 50
physical_row = logical_physical_row(0, logical_row)
physical = np.zeros((H, W), dtype=np.uint8)
physical[physical_row, col] = 1
logical = transport_manifold(physical)
reacted = inverse_transport_manifold(react_once_logical(logical))

prev_row = logical_physical_row(0, logical_row - 1)
next_row = logical_physical_row(0, logical_row + 1)
assert (physical_row - prev_row) % 64 == GEN
assert (next_row - physical_row) % 64 == GEN

expected = {
    (physical_row, col),
    (prev_row, col),
    (next_row, col),
    (physical_row, col - 1),
    (physical_row, col + 1),
}
got = set(zip(*np.nonzero(reacted)))
assert got == expected
report["G5_generator_relation"] = {
    "pass": True,
    "logical_row": logical_row,
    "physical_row": physical_row,
    "physical_prev": prev_row,
    "physical_next": next_row,
    "physical_step_mod64": GEN,
}

# G6 INFORMATION participates in REACT.
logical = np.zeros((H, W), dtype=np.uint8)
logical[31, DATA_W] = 1
reacted = react_once_logical(logical)
assert reacted[31, DATA_W - 1] == 1
report["G6_information_to_data"] = {
    "pass": True,
    "information_lane": DATA_W,
    "affected_data_lane": DATA_W - 1,
}

# G7 DATA can affect INFORMATION.
logical = np.zeros((H, W), dtype=np.uint8)
logical[31, DATA_W - 1] = 1
reacted = react_once_logical(logical)
assert reacted[31, DATA_W] == 1
report["G7_data_to_information"] = {
    "pass": True,
    "data_lane": DATA_W - 1,
    "affected_information_lane": DATA_W,
}

# G8 corrected end-to-end reference path executes with exact byte state and wide energy.
state = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
query = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
out = attention_forward(state, query, rounds=1)
assert out["state"].shape == (H, W)
assert out["state"].dtype == np.uint8
assert isinstance(out["energy"], int)
assert out["energy"] >= 0
report["G8_attention_forward"] = {
    "pass": True,
    "energy_type": "wide integer",
    "softmax": False,
    "pixel_geometry": False,
}

report["overall"] = "ALL PASS"
report["training_gate_open"] = False
report["next_open_question"] = (
    "The 1,920 INFORMATION-byte semantic codec and learned ReactionLUT policy "
    "remain open. Do not train a recognition model until those are defined/probed."
)
report["claim_boundary"] = (
    "This gate validates corrected byte layout, 64-row cache traversal, "
    "generator-7 logical transport, and DATA<->INFORMATION participation. "
    "It does not prove that the chosen 5-site REACT relation learns useful attention."
)

root = Path(__file__).resolve().parents[1]
out_path = root / "results" / "corrected_attention_survival_gate.json"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
