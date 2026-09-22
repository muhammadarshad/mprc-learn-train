"""
Arshad's ViT — reference topology v0

A ring-native vision architecture intended as an alternative to shifted-window ViT/Swin.
No NumPy, no floating point, no softmax, no learned Q/K/V matrices in this reference path.

Frozen geometry from the framework/discussion:
    R = Z_256
    image payload = 112 x 112
    QCM manifold = 128 x 113 = 14,464 sites
    structural surplus = 14,464 - 112^2 = 1,920 = 15 x 128
    128 - 112 = 16
    113 - 112 = 1
    local attention = k^2 = 3^2 = 1 reference + 8 ADI differential sites
    generator = 7 = 2^0 + 2^1 + 2^2
    chunk walk domain = 64, q_t = 7 t mod 64
    react = GEVHV 5-site staple/LUT
    measure = circular Z_256 distance energy

The 1,920 structural/header bytes are accepted as opaque caller data.  This module
specifies their exact placement/recovery but deliberately does not invent their bit
semantics.
"""
from core.constants import TAU
from attention.attention_v0 import (
    GENERATOR,
    LOCAL_KEYS,
    attention_forward,
    local_adi9,
    measure_energy,
)
from gradient.laplacian import H, W

IMAGE_H = 112
IMAGE_W = 112
MANIFOLD_H = H       # 128
MANIFOLD_W = W       # 113
MANIFOLD_SIZE = H * W
PAYLOAD_SIZE = IMAGE_H * IMAGE_W
STRUCTURAL_SIZE = MANIFOLD_SIZE - PAYLOAD_SIZE
STRUCTURAL_LANES = 15
POLARITY_WIDTH = 128
CHUNK_DOMAIN = 64

assert MANIFOLD_H - IMAGE_H == 16
assert MANIFOLD_W - IMAGE_W == 1
assert STRUCTURAL_SIZE == 1920
assert STRUCTURAL_SIZE == STRUCTURAL_LANES * POLARITY_WIDTH
assert LOCAL_KEYS == 8


def stride7_chunk_orbit(start=0):
    """Complete 64-address quantum/chunk walk q_t = start + 7t (mod 64)."""
    s = int(start) & 63
    out = []
    q = s
    for _ in range(CHUNK_DOMAIN):
        out.append(q)
        q = (q + GENERATOR) & 63
    return out


def chunk_walk2d(row_phase=0, col_phase=0):
    """Paired row/column 64-address walks with independent phases.

    This is an address schedule only; iota/phase semantics remain caller-controlled.
    """
    rows = stride7_chunk_orbit(row_phase)
    cols = stride7_chunk_orbit(col_phase)
    return list(zip(rows, cols))


def _border_indices():
    """Indices outside the 112x112 payload embedded at rows 0..111, cols 0..111.

    Geometry is exact:
      * right metadata edge: rows 0..111, col 112 -> 112 sites
      * lower structural block: rows 112..127, all 113 cols -> 1,808 sites
      total = 1,920 = 15*128.
    """
    idxs = []
    for r in range(IMAGE_H):
        idxs.append(r * MANIFOLD_W + IMAGE_W)
    for r in range(IMAGE_H, MANIFOLD_H):
        base = r * MANIFOLD_W
        for c in range(MANIFOLD_W):
            idxs.append(base + c)
    return idxs


_BORDER = _border_indices()
assert len(_BORDER) == STRUCTURAL_SIZE
assert len(set(_BORDER)) == STRUCTURAL_SIZE


def pack_image112(image, structural=None):
    """Pack a 112x112 byte image and opaque 1,920-byte structure into 128x113.

    `structural` may be either:
      * flat length-1920 iterable, or
      * 15 rows x 128 values (the transposed structural/header view).

    No metadata bits are interpreted here.
    """
    if len(image) != IMAGE_H or any(len(row) != IMAGE_W for row in image):
        raise ValueError("pack_image112: expected 112x112 image")

    if structural is None:
        meta = [0] * STRUCTURAL_SIZE
    else:
        if len(structural) == STRUCTURAL_LANES and all(
            hasattr(row, '__len__') and len(row) == POLARITY_WIDTH for row in structural
        ):
            meta = []
            for row in structural:
                meta.extend(int(x) & 0xFF for x in row)
        else:
            meta = [int(x) & 0xFF for x in structural]
        if len(meta) != STRUCTURAL_SIZE:
            raise ValueError("pack_image112: structural must contain exactly 1,920 bytes")

    out = [0] * MANIFOLD_SIZE
    for r in range(IMAGE_H):
        dst = r * MANIFOLD_W
        src = image[r]
        for c in range(IMAGE_W):
            out[dst + c] = int(src[c]) & 0xFF

    for idx, value in zip(_BORDER, meta):
        out[idx] = value
    return out


def unpack_image112(manifold):
    """Exact inverse of pack_image112 -> (112x112 image, 15x128 structure)."""
    if len(manifold) != MANIFOLD_SIZE:
        raise ValueError("unpack_image112: manifold size mismatch")

    image = []
    for r in range(IMAGE_H):
        base = r * MANIFOLD_W
        image.append([(int(manifold[base + c]) & 0xFF) for c in range(IMAGE_W)])

    flat = [(int(manifold[idx]) & 0xFF) for idx in _BORDER]
    structural = []
    off = 0
    for _ in range(STRUCTURAL_LANES):
        structural.append(flat[off:off + POLARITY_WIDTH])
        off += POLARITY_WIDTH
    return image, structural


def local_descriptor(manifold, row, col):
    """3x3 ADI attention descriptor: Lambda + eight differential keys."""
    r = int(row)
    c = int(col)
    if not (1 <= r < MANIFOLD_H - 1 and 1 <= c < MANIFOLD_W - 1):
        raise ValueError("local_descriptor: center must be interior")
    return local_adi9(manifold, r * MANIFOLD_W + c, MANIFOLD_W)


class ArshadViT:
    """Minimal executable Arshad's ViT forward topology.

    Stage 0  PACK      : 112x112 payload + 15x128 opaque structural/header state
    Stage 1  IDENTIFY  : 3x3 = 1 + (k^2-1) = 1 + 8 ADI neighborhood identity
    Stage 2  WALK      : 64-address stride-7 row/column schedule
    Stage 3  BIND      : vector query injection on Z256 (inside GEVHV fused kernel)
    Stage 4  REACT     : repeated 5-site staple + 256-byte reaction LUT
    Stage 5  MEASURE   : exact circular-distance energy in Z (ranking observable)

    The class intentionally leaves task-specific LUT learning/selection and the exact
    17-bit header codec outside v0; both can be supplied without changing the topology.
    """

    def __init__(self, lut=None, rounds=GENERATOR, row_phase=0, col_phase=0):
        self.lut = list(range(256)) if lut is None else [int(x) & 0xFF for x in lut]
        if len(self.lut) != 256:
            raise ValueError("ArshadViT: LUT must have 256 entries")
        self.rounds = int(rounds)
        if self.rounds < 1:
            raise ValueError("ArshadViT: rounds must be >= 1")
        self.walk = chunk_walk2d(row_phase, col_phase)

    def encode(self, image, structural=None):
        return pack_image112(image, structural)

    def forward_manifold(self, manifold, query):
        """Run ring-native attention on an already packed QCM manifold."""
        state, energy = attention_forward(
            manifold,
            query,
            lut=self.lut,
            rounds=self.rounds,
            gauge_each_round=False,
        )
        return {
            'state': state,
            'energy': energy,
            'chunk_walk': self.walk,
        }

    def forward(self, image, query_manifold, structural=None):
        manifold = self.encode(image, structural)
        out = self.forward_manifold(manifold, query_manifold)
        out['manifold'] = manifold
        return out

    @staticmethod
    def rank(states, query):
        """Global attention/readout: ascending exact GEVHV energy, no softmax."""
        scored = []
        for i, state in enumerate(states):
            scored.append((measure_energy(state, query), i))
        scored.sort()
        return scored