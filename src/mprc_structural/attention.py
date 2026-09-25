"""Corrected MPRC attention over the DATA + INFORMATION manifold.

This module is deliberately a reference implementation for the corrected topology.

Manifold:
    128 x 113 = 14,464 byte states
    width = 98 DATA lanes + 15 INFORMATION lanes
    execution = two 64 x 113 slabs

Generator transport:
    q_t = q_0 + 7*t (mod 64)

The transport is not merely loop order. A slab is viewed in generator order before
REACT, so logical +/-1 row neighbors correspond to physical +/-GEN movement on Z64.

Attention:
    TRANSPORT -> BIND -> REACT -> MEASURE

REACT uses the existing 5-site cross on the transported state manifold:
    center + row_prev + row_next + lane_prev + lane_next  (mod 256)

The 113 columns are state lanes, not pixels. The boundary between DATA lane 97 and
INFORMATION lane 98 is an ordinary internal manifold adjacency, so INFORMATION can
participate in attention rather than remaining dead metadata.

This file freezes only execution/topology. The semantic CONTENT of the 1,920
INFORMATION bytes and learned ReactionLUT policy remain separate/open.
"""

from __future__ import annotations

import numpy as np

from .manifold import (
    GEN,
    H,
    W,
    DATA_W,
    INFO_W,
    TILE_H,
    SLAB_COUNT,
)
from .ring import cdist

GEN_INV_64 = pow(GEN, -1, 64)
GEN_INV_256 = pow(GEN, -1, 256)

assert GEN == 7
assert GEN_INV_64 == 55
assert GEN_INV_256 == 183
assert H == 2 * TILE_H == 128
assert W == DATA_W + INFO_W == 113


def generator_orbit64(start: int = 0, generator: int = GEN) -> np.ndarray:
    """Return the full additive orbit on Z64."""
    g = int(generator) & 63
    s = int(start) & 63
    out = np.empty(TILE_H, dtype=np.int16)
    q = s
    for t in range(TILE_H):
        out[t] = q
        q = (q + g) & 63
    return out


def _validate_full_cycle(generator: int) -> np.ndarray:
    order = generator_orbit64(0, generator)
    if len(set(int(x) for x in order)) != TILE_H:
        raise ValueError("generator must have full period 64")
    return order


def transport_slab(slab: np.ndarray, generator: int = GEN) -> np.ndarray:
    """Physical 64x113 slab -> generator-logical 64x113 slab."""
    x = np.asarray(slab)
    assert x.shape == (TILE_H, W)
    assert x.dtype == np.uint8
    order = _validate_full_cycle(generator)
    return x[order, :].copy()


def inverse_transport_slab(logical: np.ndarray, generator: int = GEN) -> np.ndarray:
    """Exact inverse of transport_slab."""
    x = np.asarray(logical)
    assert x.shape == (TILE_H, W)
    assert x.dtype == np.uint8
    order = _validate_full_cycle(generator)

    out = np.empty_like(x)
    out[order, :] = x
    return out


def transport_manifold(hv: np.ndarray, generator: int = GEN) -> np.ndarray:
    """Transport both 64-row cache slabs independently."""
    x = np.asarray(hv)
    assert x.shape == (H, W)
    assert x.dtype == np.uint8

    out = np.empty_like(x)
    for s in range(SLAB_COUNT):
        lo = s * TILE_H
        out[lo:lo + TILE_H, :] = transport_slab(
            x[lo:lo + TILE_H, :], generator=generator
        )
    return out


def inverse_transport_manifold(hv: np.ndarray, generator: int = GEN) -> np.ndarray:
    """Inverse of transport_manifold."""
    x = np.asarray(hv)
    assert x.shape == (H, W)
    assert x.dtype == np.uint8

    out = np.empty_like(x)
    for s in range(SLAB_COUNT):
        lo = s * TILE_H
        out[lo:lo + TILE_H, :] = inverse_transport_slab(
            x[lo:lo + TILE_H, :], generator=generator
        )
    return out


def bind(state: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Vector BIND: Z256 addition, byte exact."""
    a = np.asarray(state)
    b = np.asarray(query)
    assert a.shape == b.shape == (H, W)
    assert a.dtype == b.dtype == np.uint8
    return ((a.astype(np.uint16) + b.astype(np.uint16)) & 0xFF).astype(np.uint8)


def identity_lut() -> np.ndarray:
    return np.arange(256, dtype=np.uint8)


def validate_lut(lut: np.ndarray | None) -> np.ndarray:
    if lut is None:
        return identity_lut()
    table = np.asarray(lut)
    assert table.shape == (256,)
    assert table.dtype == np.uint8
    return table


def react_once_logical(logical: np.ndarray, lut: np.ndarray | None = None) -> np.ndarray:
    """One exact 5-site REACT round in generator-logical coordinates.

    Boundary remains identity, matching the existing v1 rule.
    """
    x = np.asarray(logical)
    assert x.shape == (H, W)
    assert x.dtype == np.uint8
    table = validate_lut(lut)

    out = x.copy()

    # Process each 64-row slab independently so one 64x113 slab is the execution unit.
    # Interior logical rows 1..62 and lanes 1..111 use the five-site relation.
    for s in range(SLAB_COUNT):
        lo = s * TILE_H
        hi = lo + TILE_H
        slab = x[lo:hi, :]

        acc = (
            slab[1:-1, 1:-1].astype(np.uint16)
            + slab[:-2, 1:-1].astype(np.uint16)
            + slab[2:, 1:-1].astype(np.uint16)
            + slab[1:-1, :-2].astype(np.uint16)
            + slab[1:-1, 2:].astype(np.uint16)
        ) & 0xFF

        out[lo + 1:hi - 1, 1:-1] = table[acc.astype(np.uint8)]

    return out


def react(
    logical: np.ndarray,
    lut: np.ndarray | None = None,
    rounds: int = 1,
) -> np.ndarray:
    """Repeated REACT in transported coordinates."""
    r = int(rounds)
    if r < 1:
        raise ValueError("rounds must be >= 1")
    state = np.asarray(logical)
    assert state.shape == (H, W)
    assert state.dtype == np.uint8
    state = state.copy()
    for _ in range(r):
        state = react_once_logical(state, lut=lut)
    return state


def measure_energy(state: np.ndarray, query: np.ndarray) -> int:
    """Exact wide integer sum of circular Z256 distance."""
    a = np.asarray(state)
    b = np.asarray(query)
    assert a.shape == b.shape == (H, W)
    assert a.dtype == b.dtype == np.uint8

    # Vectorized exact equivalent of ring.cdist.
    ab = (a.astype(np.int16) - b.astype(np.int16)) & 0xFF
    ba = (b.astype(np.int16) - a.astype(np.int16)) & 0xFF
    d = np.minimum(ab, ba)
    return int(d.sum(dtype=np.int64))


def attention_forward(
    state: np.ndarray,
    query: np.ndarray,
    *,
    lut: np.ndarray | None = None,
    rounds: int = 1,
    generator: int = GEN,
    return_physical_state: bool = True,
) -> dict:
    """Corrected reference attention.

    The same generator transport is applied to state and query. BIND and REACT happen
    in logical generator coordinates. MEASURE compares in that same coordinate system.
    Optionally return the reacted state mapped back to physical manifold coordinates.
    """
    a = np.asarray(state)
    q = np.asarray(query)
    assert a.shape == q.shape == (H, W)
    assert a.dtype == q.dtype == np.uint8

    ta = transport_manifold(a, generator=generator)
    tq = transport_manifold(q, generator=generator)

    bound = ((ta.astype(np.uint16) + tq.astype(np.uint16)) & 0xFF).astype(np.uint8)
    reacted = react(bound, lut=lut, rounds=rounds)
    energy = measure_energy(reacted, tq)

    result = {
        "energy": energy,
        "logical_state": reacted,
        "generator": int(generator),
        "rounds": int(rounds),
    }
    if return_physical_state:
        result["state"] = inverse_transport_manifold(reacted, generator=generator)
    return result


def logical_physical_row(slab_index: int, logical_row: int, generator: int = GEN) -> int:
    """Map one logical row in a 64-row slab back to physical manifold row."""
    s = int(slab_index)
    t = int(logical_row)
    if not 0 <= s < SLAB_COUNT:
        raise ValueError("slab_index must be 0 or 1")
    if not 0 <= t < TILE_H:
        raise ValueError("logical_row must be 0..63")
    order = _validate_full_cycle(generator)
    return s * TILE_H + int(order[t])
