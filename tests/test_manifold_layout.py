import numpy as np

from mprc_structural.manifold import (
    CHANNEL_COUNT,
    CHANNEL_NAMES,
    MANIFOLD_H,
    MANIFOLD_W,
    PAYLOAD_H,
    PAYLOAD_W,
    StructuralMetadata,
    layout_counts,
    pack,
    unpack,
)


def test_frozen_dimensions():
    assert (MANIFOLD_H, MANIFOLD_W) == (128, 113)
    assert (PAYLOAD_H, PAYLOAD_W) == (112, 112)
    assert CHANNEL_COUNT == 16
    assert len(CHANNEL_NAMES) == 16


def test_storage_arithmetic():
    c = layout_counts()
    assert c["payload"] == 112 * 112 == 12544
    assert c["channel_storage"] == 16 * 112 == 1792
    assert c["polarity_storage"] == 112
    assert c["channel_polarity_corner"] == 16
    assert c["structural_total"] == 1920
    assert c["manifold_total"] == 128 * 113 == 14464
    assert c["payload"] + c["structural_total"] == c["manifold_total"]


def test_lossless_payload_and_metadata_roundtrip():
    rng = np.random.default_rng(20260923)

    payload = rng.integers(0, 256, size=(112, 112), dtype=np.uint8)
    metadata = StructuralMetadata(
        channel_rows=rng.integers(0, 256, size=(16, 112), dtype=np.uint8),
        polarity_column=rng.integers(0, 256, size=(112,), dtype=np.uint8),
        channel_polarity_corner=rng.integers(0, 256, size=(16,), dtype=np.uint8),
    )

    hv = pack(payload, metadata)
    out_payload, out_meta = unpack(hv)

    assert np.array_equal(out_payload, payload)
    assert np.array_equal(out_meta.channel_rows, metadata.channel_rows)
    assert np.array_equal(out_meta.polarity_column, metadata.polarity_column)
    assert np.array_equal(
        out_meta.channel_polarity_corner,
        metadata.channel_polarity_corner,
    )
