from mprc_structural.block_local112 import (
    BLOCK_STATES,
    block_h_to_v,
    block_v_to_h,
    decode_h,
    decode_v,
    encode_h,
    encode_v,
    local_h_to_v,
    local_v_to_h,
    transport_block_polarity_h_to_v,
    transport_block_polarity_v_to_h,
    transport_channel_positions_h_to_v,
    transport_channel_positions_v_to_h,
)


def test_pixel_factorization_is_bijective():
    seen_h = set()
    seen_v = set()

    for r in range(112):
        for c in range(112):
            h = encode_h(r, c)
            v = encode_v(r, c)

            assert h not in seen_h
            assert v not in seen_v
            seen_h.add(h)
            seen_v.add(v)

            assert decode_h(*h) == (r, c)
            assert decode_v(*v) == (r, c)

    assert len(seen_h) == 112 * 112
    assert len(seen_v) == 112 * 112


def test_transpose_transport_matches_pixel_transpose():
    for r in range(112):
        for c in range(112):
            hb, hl = encode_h(r, c)

            vb = block_h_to_v(hb)
            vl = local_h_to_v(hl)

            assert (vb, vl) == encode_v(c, r)


def test_112_state_permutations_are_bijective():
    assert len({block_h_to_v(x) for x in range(BLOCK_STATES)}) == 112
    assert len({local_h_to_v(x) for x in range(BLOCK_STATES)}) == 112

    for x in range(BLOCK_STATES):
        assert block_v_to_h(block_h_to_v(x)) == x
        assert local_v_to_h(local_h_to_v(x)) == x


def test_channel_position_transport_roundtrip():
    positions = [
        [((17 * channel) + block) % 112 for block in range(112)]
        for channel in range(16)
    ]

    v = transport_channel_positions_h_to_v(positions)
    h = transport_channel_positions_v_to_h(v)

    assert h == positions


def test_block_polarity_transport_roundtrip():
    polarity = [(13 * block) & 0xFF for block in range(112)]

    v = transport_block_polarity_h_to_v(polarity)
    h = transport_block_polarity_v_to_h(v)

    assert h == polarity
