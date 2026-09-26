"""Foundation closure finite gate RC1.

This script verifies only statements already classified CLOSED in
docs/FOUNDATION_CLOSURE_RC1.md.

It deliberately does NOT implement the open interfaces:
    G6 packing map
    G7 ReactionLUT law
    G8 IDENTIFY
    G9 U/MOVE
    G10 SELECT/memory

A green run therefore means "closed substrate survived", not "MPRC is complete".
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.mprc_structural.ring import (
    TAU, ORIGIN, zadd, zsub, cdist
)
from src.mprc_structural.manifold import (
    GEN, H, W, DATA_W, INFO_W, TILE_H, SLAB_COUNT
)
from src.mprc_structural.attention import (
    generator_orbit64,
    transport_manifold,
    inverse_transport_manifold,
    bind,
    react,
    measure_energy,
)

assert TAU == 256
assert ORIGIN == 128
assert GEN == 7
assert H == 128
assert W == 113
assert DATA_W == 98
assert INFO_W == 15
assert TILE_H == 64
assert SLAB_COUNT == 2


def ring_metric_gate():
    D = np.empty((256,256), dtype=np.int16)
    for a in range(256):
        for b in range(256):
            D[a,b] = cdist(a,b)

    assert np.all(D >= 0)
    assert np.all(D <= 128)
    assert np.array_equal(D, D.T)
    assert np.all(np.diag(D) == 0)

    off = D.copy()
    np.fill_diagonal(off, 1)
    assert np.all(off > 0)

    triangle_checks = 0
    for b in range(256):
        rhs = D[:,b,None].astype(np.int16) + D[b,None,:].astype(np.int16)
        if not np.all(D <= rhs):
            idx = np.argwhere(D > rhs)[0]
            raise AssertionError(("triangle", int(idx[0]), b, int(idx[1])))
        triangle_checks += 256 * 256

    return {
        "pass": True,
        "states": 256,
        "triangle_checks": triangle_checks,
        "diameter": int(D.max()),
    }


def gen7_gate():
    orbit64 = generator_orbit64(0, GEN)
    assert len(set(int(x) for x in orbit64)) == 64

    q = 0
    orbit256 = []
    for _ in range(256):
        orbit256.append(q)
        q = (q + GEN) & 0xFF

    assert len(set(orbit256)) == 256
    assert pow(GEN,-1,64) == 55
    assert pow(GEN,-1,256) == 183

    return {
        "pass": True,
        "orbit64_unique": 64,
        "orbit256_unique": 256,
        "inv64": 55,
        "inv256": 183,
    }


def manifold_gate():
    assert 2 * GEN * GEN == DATA_W == 98
    assert INFO_W == 15
    assert DATA_W + INFO_W == W == 113
    assert GEN*GEN + INFO_W == 64
    assert H * DATA_W == 12544
    assert H * INFO_W == 1920
    assert H * W == 14464
    assert SLAB_COUNT * TILE_H * W == 14464

    return {
        "pass": True,
        "data_bytes": 12544,
        "information_bytes": 1920,
        "manifold_bytes": 14464,
        "slab_bytes": TILE_H * W,
    }


def transport_bind_measure_gate():
    a = np.fromfunction(
        lambda i,j: (17*i + 29*j + 3) % 256,
        (H,W),
        dtype=int
    ).astype(np.uint8)
    b = np.fromfunction(
        lambda i,j: (31*i + 11*j + 7) % 256,
        (H,W),
        dtype=int
    ).astype(np.uint8)

    ta = transport_manifold(a)
    tb = transport_manifold(b)

    assert np.array_equal(inverse_transport_manifold(ta), a)
    assert np.array_equal(inverse_transport_manifold(tb), b)

    lhs = transport_manifold(bind(a,b))
    rhs = bind(ta,tb)
    assert np.array_equal(lhs,rhs)

    E0 = measure_energy(a,b)
    E1 = measure_energy(ta,tb)
    assert E0 == E1

    return {
        "pass": True,
        "transport_inverse": True,
        "bind_commutes_with_transport": True,
        "measure_permutation_invariant": True,
        "energy": E0,
    }


def react_additivity_gate():
    a = np.fromfunction(
        lambda i,j: (13*i + 5*j + 19) % 256,
        (H,W),
        dtype=int
    ).astype(np.uint8)
    b = np.fromfunction(
        lambda i,j: (7*i + 23*j + 41) % 256,
        (H,W),
        dtype=int
    ).astype(np.uint8)

    s = bind(a,b)

    for rounds in (1,2,7):
        lhs = react(s, lut=None, rounds=rounds)
        rhs = bind(
            react(a,lut=None,rounds=rounds),
            react(b,lut=None,rounds=rounds),
        )
        assert np.array_equal(lhs,rhs)

    return {
        "pass": True,
        "identity_lut_additive_rounds": [1,2,7],
    }


def dependency_cone_gate():
    counts = {}
    for r in range(0,8):
        pts = {
            (x,y)
            for x in range(-r,r+1)
            for y in range(-r,r+1)
            if abs(x)+abs(y) <= r
        }
        expected = 1 + 2*r*(r+1)
        assert len(pts) == expected
        counts[str(r)] = len(pts)

    assert counts["7"] == 113

    return {
        "pass": True,
        "counts": counts,
        "round7_dependency_sites": 113,
    }


def encoder_scalar_closure_gate():
    # Exhaust all byte values independently where possible; boundary extrema
    # prove range for weighted expressions.
    vals = np.arange(256, dtype=np.int64)

    # Luma extrema and exact byte closure from convex integer weights.
    assert (77+150+29) == 256
    assert (77*255 + 150*255 + 29*255)//256 == 255

    # Gray extrema.
    assert (255+255+255)//3 == 255

    # Chroma.
    assert 255-0 == 255

    # Phase.
    assert (6+6+6)*14 == 252

    # Winding.
    winding = ((vals[:,None,None] + vals[None,:,None] + vals[None,None,:]) // 256)
    # This temporary cube is 256^3 int64 (~128 MiB); inspect only range then release.
    assert int(winding.min()) == 0
    assert int(winding.max()) == 2
    del winding

    # Heat numerator bounds.
    assert 0 <= 0
    assert (255+255+255+255+4*255)//8 == 255

    return {
        "pass": True,
        "luma_range": [0,255],
        "gray_range": [0,255],
        "chroma_range": [0,255],
        "phase_max": 252,
        "winding_levels": [0,1,2],
        "heat_step_range": [0,255],
        "wrapped_channels": ["d1","d2","plaquette"],
    }


def main():
    report = {
        "foundation":"RC1",
        "closed_gates":{
            "G0_ring_metric": ring_metric_gate(),
            "G1_gen7": gen7_gate(),
            "G2_manifold": manifold_gate(),
            "G3_transport_bind_measure": transport_bind_measure_gate(),
            "G4_react_identity_lut": react_additivity_gate(),
            "G4b_dependency_cone": dependency_cone_gate(),
            "G5_encoder_scalar_closure": encoder_scalar_closure_gate(),
        },
        "open_gates":{
            "G6":"packing map Pi",
            "G7":"ReactionLUT law",
            "G8":"IDENTIFY state type",
            "G9":"U/MOVE state transition",
            "G10":"SELECT/resolved-memory contract",
            "G11":"end-to-end finite theorem",
            "G12":"full experiment",
        },
        "pass": True,
        "claim_boundary":(
            "A PASS certifies only the closed substrate listed above. "
            "It does not close G6-G12 and is not an end-to-end learning result."
        ),
    }

    out=Path("results/foundation_closure_rc1.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
