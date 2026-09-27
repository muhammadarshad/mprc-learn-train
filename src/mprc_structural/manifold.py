"""Canonical MPRC DATA + INFORMATION computational manifold.

Frozen algebra:
    D = 256
    N = sqrt(D) - 1 = 15
    GEN = sqrt(D)/2 - 1 = 7
    H = D/2 = 128
    W_DATA = 2*GEN^2 = 98
    W_INFO = N = 15
    W = 113

    DATA STATES        = 128*98  = 12,544
    INFORMATION STATES = 128*15  = 1,920
    MANIFOLD STATES    = 128*113 = 14,464

These are computational Z256 state/address slots in the frozen ViT algebra.
They are NOT a claim about source-image pixel count or physical packed storage
capacity.  Source pixels, materialized computational arrays, and physical
storage accounting must be reported separately.

Execution remains two 64x113 slabs with generator-7 transport on Z64.
No 16x113/Z16 execution reinterpretation is canonical.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

D = 256
ROOT_D = math.isqrt(D)
assert ROOT_D * ROOT_D == D

N = ROOT_D - 1                    # 15
GEN = ROOT_D // 2 - 1            # 7
H = D // 2                       # 128

DATA_W = 2 * GEN * GEN           # 98
INFO_W = N                       # 15
W = DATA_W + INFO_W              # 113

DATA_STATES = H * DATA_W         # 12,544
INFO_STATES = H * INFO_W         # 1,920
MANIFOLD_STATES = H * W          # 14,464

TILE_H = 64
SLAB_COUNT = H // TILE_H         # 2
SLAB_STATES = TILE_H * W         # 7,232
SLAB_DATA_STATES = TILE_H * DATA_W  # 6,272
SLAB_INFO_STATES = TILE_H * INFO_W  # 960

# Independent closure identities.
assert GEN * GEN + N == D // 4 == 64
assert DATA_STATES == 49 * 256 == 98 * 128
assert INFO_STATES == 15 * 128
assert MANIFOLD_STATES == DATA_STATES + INFO_STATES == 113 * 128
assert H == 2 * TILE_H
assert SLAB_COUNT == 2

# Backward-compatibility aliases for older experiments/tests.  The names are
# retained only to avoid breaking code; semantically these count materialized
# computational Z256 state slots, not packed physical bytes.
DATA_BYTES = DATA_STATES
INFO_BYTES = INFO_STATES
MANIFOLD_BYTES = MANIFOLD_STATES
SLAB_BYTES = SLAB_STATES
SLAB_DATA_BYTES = SLAB_DATA_STATES
SLAB_INFO_BYTES = SLAB_INFO_STATES


@dataclass(frozen=True)
class ManifoldState:
    """Modality-agnostic payload for the canonical manifold.

    data:
        Flat uint8 vector of exactly 12,544 materialized DATA states.

    information:
        Flat uint8 vector of exactly 1,920 materialized INFORMATION states.

    No pixel, image, channel-row, or polarity-column semantics are imposed.
    """

    data: np.ndarray
    information: np.ndarray

    def validate(self) -> None:
        assert self.data.shape == (DATA_BYTES,)
        assert self.information.shape == (INFO_BYTES,)
        assert self.data.dtype == np.uint8
        assert self.information.dtype == np.uint8


def empty_state() -> ManifoldState:
    return ManifoldState(
        data=np.zeros(DATA_BYTES, dtype=np.uint8),
        information=np.zeros(INFO_BYTES, dtype=np.uint8),
    )


def pack(data: np.ndarray, information: np.ndarray) -> np.ndarray:
    """Pack DATA + INFORMATION into the canonical 128x113 state manifold.

    Physical column partition:
        cols 0..97   = DATA     (98 lanes)
        cols 98..112 = INFO     (15 lanes)

    This partition follows W = 98 + 15. It is a state layout, not image geometry.
    """
    state = ManifoldState(np.asarray(data), np.asarray(information))
    state.validate()

    hv = np.empty((H, W), dtype=np.uint8)
    hv[:, :DATA_W] = state.data.reshape(H, DATA_W)
    hv[:, DATA_W:] = state.information.reshape(H, INFO_W)
    return hv


def unpack(hv: np.ndarray) -> ManifoldState:
    """Exact inverse of pack()."""
    hv = np.asarray(hv)
    assert hv.shape == (H, W)
    assert hv.dtype == np.uint8

    return ManifoldState(
        data=hv[:, :DATA_W].reshape(DATA_BYTES).copy(),
        information=hv[:, DATA_W:].reshape(INFO_BYTES).copy(),
    )


def slab(hv: np.ndarray, slab_index: int) -> np.ndarray:
    """Return one contiguous logical 64x113 execution slab."""
    hv = np.asarray(hv)
    assert hv.shape == (H, W)
    assert hv.dtype == np.uint8
    s = int(slab_index)
    if not 0 <= s < SLAB_COUNT:
        raise ValueError("slab_index must be 0 or 1")
    lo = s * TILE_H
    return hv[lo:lo + TILE_H, :]


def layout_counts() -> dict[str, int]:
    return {
        "D": D,
        "sqrt_D": ROOT_D,
        "N": N,
        "GEN": GEN,
        "H": H,
        "data_width": DATA_W,
        "information_width": INFO_W,
        "W": W,
        "data_states": DATA_STATES,
        "data": DATA_STATES,
        "information_states": INFO_STATES,
        "information": INFO_STATES,
        "manifold_states": MANIFOLD_STATES,
        "manifold_total": MANIFOLD_STATES,
        "slab_count": SLAB_COUNT,
        "slab_bytes": SLAB_BYTES,
        "slab_data_bytes": SLAB_DATA_BYTES,
        "slab_information_bytes": SLAB_INFO_BYTES,
    }
