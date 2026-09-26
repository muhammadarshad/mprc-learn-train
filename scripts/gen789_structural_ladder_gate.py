"""GEN 7-8-9 structural ladder finite gate.

Verifies the exact identities frozen in:
    docs/GEN789_STRUCTURAL_LADDER_RC1.md

This is a math gate, not a model benchmark.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path


def ladder_from_D(D: int):
    root = int(D ** 0.5)
    assert root * root == D
    assert root % 2 == 0
    s = root // 2
    g = s - 1
    N = root - 1
    W = 2 * g * g + N
    active = 4 * g * (g + 2)
    return {
        "D": D,
        "sqrtD": root,
        "s": s,
        "GEN": g,
        "N": N,
        "W": W,
        "ACTIVE": active,
    }


def frozen_256_gate():
    x = ladder_from_D(256)

    assert x["sqrtD"] == 16
    assert x["s"] == 8
    assert x["GEN"] == 7
    assert x["N"] == 15

    g = x["GEN"]
    N = x["N"]

    assert N == 2 * g + 1

    W1 = 2 * g * g + N
    W2 = 1 + 2 * g * (g + 1)
    W3 = 256 // 2 - 16 + 1
    manhattan = 1 + 2 * g * (g + 1)

    assert W1 == W2 == W3 == manhattan == 113

    A1 = 4 * g * (g + 2)
    A2 = 4 * ((g + 1) * (g + 1) - 1)
    A3 = 256 - 4

    assert A1 == A2 == A3 == 252
    assert 4 * (3 + 4) * 9 == 252

    return {
        "pass": True,
        "ladder": [g, g + 1, g + 2],
        "N": N,
        "W_forms": [W1, W2, W3, manhattan],
        "ACTIVE_forms": [A1, A2, A3],
    }


def manhattan_ball_gate(max_r: int = 32):
    rows = []
    for r in range(max_r + 1):
        pts = {
            (x, y)
            for x in range(-r, r + 1)
            for y in range(-r, r + 1)
            if abs(x) + abs(y) <= r
        }
        formula = 1 + 2 * r * (r + 1)
        assert len(pts) == formula
        rows.append((r, formula))

    assert dict(rows)[7] == 113

    return {
        "pass": True,
        "radii_checked": max_r + 1,
        "radius7": 113,
    }


def fixed_phase_gate():
    """Exhaust the only inputs Phase depends on: residues mod 7."""

    vals = []
    for r, g, b in itertools.product(range(7), repeat=3):
        phase = (r + g + b) * 14
        vals.append(phase)

    assert min(vals) == 0
    assert max(vals) == (6 + 6 + 6) * 14 == 252
    assert max(vals) == 4 * 7 * 9
    assert all(0 <= v <= 255 for v in vals)

    distinct = sorted(set(vals))
    assert distinct == [14 * k for k in range(19)]

    return {
        "pass": True,
        "residue_combinations": 7 ** 3,
        "phase_min": 0,
        "phase_max": 252,
        "phase_distinct_states": len(distinct),
        "phase_values": distinct,
    }


def generalized_identity_gate(max_g: int = 1024):
    """Verify the ladder identities for the induced family D(g)=4(g+1)^2."""

    checked = 0
    for g in range(1, max_g + 1):
        D = 4 * (g + 1) * (g + 1)
        root = 2 * (g + 1)
        N = root - 1

        W_manifold = 2 * g * g + N
        W_ball = 1 + 2 * g * (g + 1)
        W_D = D // 2 - root + 1

        active_ladder = 4 * g * (g + 2)
        active_D = D - 4

        assert N == 2 * g + 1
        assert W_manifold == W_ball == W_D
        assert active_ladder == active_D
        checked += 1

    return {
        "pass": True,
        "positive_generators_checked": checked,
        "range": [1, max_g],
    }


def conditional_uniqueness_gate(max_g: int = 4096):
    """Candidate family Phi_g = (sum RGB mod g)*(2g).

    Its maximum is 6g(g-1).  Check where it equals 4g(g+2).
    """
    matches = []

    for g in range(1, max_g + 1):
        phase_max = 6 * g * (g - 1)
        active = 4 * g * (g + 2)
        if phase_max == active:
            matches.append(g)

    assert matches == [7]

    # Algebraic residual factorization:
    # 6g(g-1) - 4g(g+2) = 2g(g-7)
    for g in range(1, max_g + 1):
        lhs = 6 * g * (g - 1) - 4 * g * (g + 2)
        rhs = 2 * g * (g - 7)
        assert lhs == rhs

    D = 4 * (7 + 1) ** 2
    assert D == 256

    return {
        "pass": True,
        "candidate_family": "Phi_g=(sum of three residues mod g)*(2g)",
        "searched_positive_g": [1, max_g],
        "matches": matches,
        "factorization": "6g(g-1)-4g(g+2)=2g(g-7)",
        "selected_D": D,
    }


def status_boundary():
    return {
        "exact": [
            "N=2GEN+1",
            "W=2GEN^2+N=1+2GEN(GEN+1)",
            "W=|B_GEN| for the five-site Manhattan dependency geometry",
            "4GEN(GEN+2)=D-4",
            "(6+6+6)*14=4*7*9=252 for the fixed encoder",
        ],
        "candidate": [
            "Phi_g=(sum of three residues mod g)*(2g)",
        ],
        "conditional_theorem": [
            "Phi_max(g)=4g(g+2) iff g=7 for positive integer g",
        ],
        "not_proved": [
            "Phase semantics are identical to ACTIVE-state semantics",
            "the 7-8-9 ladder closes the observation-to-structural packing map",
        ],
    }


def main():
    report = {
        "name": "GEN789 Structural Ladder RC1",
        "frozen_256": frozen_256_gate(),
        "manhattan": manhattan_ball_gate(),
        "fixed_phase": fixed_phase_gate(),
        "generalized_ladder": generalized_identity_gate(),
        "conditional_uniqueness": conditional_uniqueness_gate(),
        "status_boundary": status_boundary(),
        "pass": True,
    }

    out = Path("results/gen789_structural_ladder_rc1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
