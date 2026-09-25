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
    GEN,
    H,
    W,
    DATA_W,
    INFO_W,
    TILE_H,
)


def test_generator_orbit_and_inverses():
    orbit = generator_orbit64()
    assert len(orbit) == 64
    assert len(set(map(int, orbit))) == 64
    assert GEN == 7
    assert GEN_INV_64 == 55
    assert GEN_INV_256 == 183
    assert (GEN * GEN_INV_64) % 64 == 1
    assert (GEN * GEN_INV_256) % 256 == 1


def test_transport_roundtrip_exact():
    rng = np.random.default_rng(20260925)
    x = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    y = transport_manifold(x)
    z = inverse_transport_manifold(y)
    assert np.array_equal(z, x)


def test_generator_changes_physical_react_neighborhood():
    # Put an impulse at an interior logical row after transport.
    logical_row = 31
    col = 50
    physical_row = logical_physical_row(0, logical_row)

    physical = np.zeros((H, W), dtype=np.uint8)
    physical[physical_row, col] = 1

    logical = transport_manifold(physical)
    assert logical[logical_row, col] == 1

    reacted_logical = react_once_logical(logical)
    reacted_physical = inverse_transport_manifold(reacted_logical)

    prev_row = logical_physical_row(0, logical_row - 1)
    next_row = logical_physical_row(0, logical_row + 1)

    # +/-1 in logical generator order is +/-7 in physical Z64 row address.
    assert (physical_row - prev_row) % 64 == GEN
    assert (next_row - physical_row) % 64 == GEN

    expected = {
        (physical_row, col),
        (prev_row, col),
        (next_row, col),
        (physical_row, col - 1),
        (physical_row, col + 1),
    }
    got = set(zip(*np.nonzero(reacted_physical)))
    assert got == expected


def test_information_is_live_attention_state():
    # Lane 97 is final DATA lane; lane 98 is first INFORMATION lane.
    assert DATA_W == 98
    assert INFO_W == 15

    logical = np.zeros((H, W), dtype=np.uint8)
    logical[31, DATA_W] = 1  # first INFORMATION lane

    reacted = react_once_logical(logical)

    # The information state directly participates in the 5-site relation of
    # the adjacent data lane.
    assert reacted[31, DATA_W - 1] == 1


def test_data_can_affect_information():
    logical = np.zeros((H, W), dtype=np.uint8)
    logical[31, DATA_W - 1] = 1  # final DATA lane

    reacted = react_once_logical(logical)
    assert reacted[31, DATA_W] == 1


def test_two_slabs_preserve_every_state():
    ids = np.arange(H * W, dtype=np.uint16).reshape(H, W)
    # Compare coordinate visitation rather than storing >255 IDs as ring states.
    orbit = generator_orbit64()
    seen = []
    for slab_index in range(2):
        base = slab_index * TILE_H
        for logical_row in range(TILE_H):
            physical_row = base + int(orbit[logical_row])
            for w in range(W):
                seen.append((physical_row, w))
    assert len(seen) == H * W == 14_464
    assert len(set(seen)) == H * W


def test_attention_forward_is_integer_ring_path():
    rng = np.random.default_rng(19)
    state = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    query = rng.integers(0, 256, size=(H, W), dtype=np.uint8)

    out = attention_forward(state, query, rounds=1)
    assert isinstance(out["energy"], int)
    assert out["energy"] >= 0
    assert out["state"].shape == (H, W)
    assert out["state"].dtype == np.uint8
