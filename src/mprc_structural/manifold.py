"""Canonical Arshad-ViT / MPRC manifold storage layout.

The physical contract is frozen as:

    (112 + 16 channels) x (112 + 1 polarity)
      = 128 x 113
      = 14,464 uint8 states

Payload pixels are preserved exactly in the upper-left 112x112 region.
Metadata occupies the remaining 1,920 states.

This module freezes WHERE metadata lives, not WHAT every metadata byte means.
The detailed channel/polarity codec is deliberately left separate.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

PAYLOAD_H = 112
PAYLOAD_W = 112
CHANNEL_COUNT = 16
POLARITY_COUNT = 1

MANIFOLD_H = PAYLOAD_H + CHANNEL_COUNT   # 128
MANIFOLD_W = PAYLOAD_W + POLARITY_COUNT # 113

PAYLOAD_STATES = PAYLOAD_H * PAYLOAD_W          # 12,544
MANIFOLD_STATES = MANIFOLD_H * MANIFOLD_W       # 14,464
STRUCTURAL_STATES = MANIFOLD_STATES - PAYLOAD_STATES # 1,920

CHANNEL_NAMES = (
    "R", "G", "B", "gray", "luma", "chroma",
    "gx", "gy", "grad", "lap",
    "h1", "h2", "m4", "l8", "contrast", "orient",
)


@dataclass(frozen=True)
class StructuralMetadata:
    """Opaque metadata payload for the frozen physical regions.

    No semantics beyond shape are imposed here.

    channel_rows:
        16 x 112 states, one storage row per channel.

    polarity_column:
        112 states, the single polarity storage dimension.

    channel_polarity_corner:
        16 states at the intersection of channel rows and polarity column.
    """

    channel_rows: np.ndarray
    polarity_column: np.ndarray
    channel_polarity_corner: np.ndarray

    def validate(self) -> None:
        assert self.channel_rows.shape == (CHANNEL_COUNT, PAYLOAD_W)
        assert self.polarity_column.shape == (PAYLOAD_H,)
        assert self.channel_polarity_corner.shape == (CHANNEL_COUNT,)
        assert self.channel_rows.dtype == np.uint8
        assert self.polarity_column.dtype == np.uint8
        assert self.channel_polarity_corner.dtype == np.uint8


def empty_metadata() -> StructuralMetadata:
    return StructuralMetadata(
        channel_rows=np.zeros((CHANNEL_COUNT, PAYLOAD_W), dtype=np.uint8),
        polarity_column=np.zeros((PAYLOAD_H,), dtype=np.uint8),
        channel_polarity_corner=np.zeros((CHANNEL_COUNT,), dtype=np.uint8),
    )


def pack(payload: np.ndarray, metadata: StructuralMetadata) -> np.ndarray:
    """Losslessly pack 112x112 payload and structural storage into 128x113."""
    assert payload.shape == (PAYLOAD_H, PAYLOAD_W)
    assert payload.dtype == np.uint8
    metadata.validate()

    hv = np.zeros((MANIFOLD_H, MANIFOLD_W), dtype=np.uint8)

    # Pixels are never replaced by metadata.
    hv[:PAYLOAD_H, :PAYLOAD_W] = payload

    # 16 channel storage rows.
    hv[PAYLOAD_H:, :PAYLOAD_W] = metadata.channel_rows

    # One polarity storage column over payload rows.
    hv[:PAYLOAD_H, PAYLOAD_W] = metadata.polarity_column

    # 16 x 1 channel/polarity intersection.
    hv[PAYLOAD_H:, PAYLOAD_W] = metadata.channel_polarity_corner

    return hv


def unpack(hv: np.ndarray) -> tuple[np.ndarray, StructuralMetadata]:
    """Recover payload and all structural storage without loss."""
    assert hv.shape == (MANIFOLD_H, MANIFOLD_W)
    assert hv.dtype == np.uint8

    payload = hv[:PAYLOAD_H, :PAYLOAD_W].copy()
    metadata = StructuralMetadata(
        channel_rows=hv[PAYLOAD_H:, :PAYLOAD_W].copy(),
        polarity_column=hv[:PAYLOAD_H, PAYLOAD_W].copy(),
        channel_polarity_corner=hv[PAYLOAD_H:, PAYLOAD_W].copy(),
    )
    metadata.validate()
    return payload, metadata


def layout_counts() -> dict[str, int]:
    return {
        "payload": PAYLOAD_STATES,
        "channel_storage": CHANNEL_COUNT * PAYLOAD_W,
        "polarity_storage": PAYLOAD_H * POLARITY_COUNT,
        "channel_polarity_corner": CHANNEL_COUNT * POLARITY_COUNT,
        "structural_total": STRUCTURAL_STATES,
        "manifold_total": MANIFOLD_STATES,
    }
