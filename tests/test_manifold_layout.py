import numpy as np

from mprc_structural.manifold import (
    D,
    ROOT_D,
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
    SLAB_COUNT,
    SLAB_BYTES,
    SLAB_DATA_BYTES,
    SLAB_INFO_BYTES,
    ManifoldState,
    layout_counts,
    pack,
    slab,
    unpack,
)


def test_canonical_dimensions_from_D():
    assert D == 256
    assert ROOT_D == 16
    assert N == 15
    assert GEN == 7
    assert H == 128
    assert DATA_W == 98 == 2 * GEN * GEN
    assert INFO_W == 15 == N
    assert W == 113 == DATA_W + INFO_W
    assert GEN * GEN + N == 64 == D // 4


def test_storage_arithmetic():
    c = layout_counts()
    assert c["data"] == DATA_BYTES == 12_544
    assert c["information"] == INFO_BYTES == 1_920
    assert c["manifold_total"] == MANIFOLD_BYTES == 14_464
    assert c["data"] + c["information"] == c["manifold_total"]

    assert DATA_BYTES == 49 * 256
    assert DATA_BYTES == 98 * 128
    assert INFO_BYTES == 15 * 128
    assert MANIFOLD_BYTES == 113 * 128


def test_two_cache_slabs():
    assert H == 2 * TILE_H
    assert TILE_H == 64
    assert SLAB_COUNT == 2
    assert SLAB_BYTES == 64 * 113 == 7_232
    assert SLAB_DATA_BYTES == 64 * 98 == 6_272
    assert SLAB_INFO_BYTES == 64 * 15 == 960
    assert 2 * SLAB_BYTES == MANIFOLD_BYTES


def test_lossless_data_information_roundtrip():
    rng = np.random.default_rng(20260925)

    data = rng.integers(0, 256, size=DATA_BYTES, dtype=np.uint8)
    information = rng.integers(0, 256, size=INFO_BYTES, dtype=np.uint8)

    hv = pack(data, information)
    assert hv.shape == (H, W)

    out = unpack(hv)
    assert isinstance(out, ManifoldState)
    assert np.array_equal(out.data, data)
    assert np.array_equal(out.information, information)

    assert np.array_equal(hv[:, :DATA_W].reshape(-1), data)
    assert np.array_equal(hv[:, DATA_W:].reshape(-1), information)


def test_slab_partition_is_exact():
    rng = np.random.default_rng(8)
    data = rng.integers(0, 256, size=DATA_BYTES, dtype=np.uint8)
    information = rng.integers(0, 256, size=INFO_BYTES, dtype=np.uint8)
    hv = pack(data, information)

    s0 = slab(hv, 0)
    s1 = slab(hv, 1)

    assert s0.shape == (64, 113)
    assert s1.shape == (64, 113)
    assert np.array_equal(np.concatenate([s0, s1], axis=0), hv)
