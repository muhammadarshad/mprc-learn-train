"""Exact 112-state block/local factorization for Arshad-ViT geometry.

A 112x112 payload in the H phase is partitioned into:
    7 x 16 blocks, each block 16 x 7.

Therefore:
    number of blocks          = 7*16  = 112
    positions inside a block = 16*7  = 112

Every pixel is identified bijectively by:
    (block_index, local_position) in [0,111]^2.

Under the exact 16x7 -> 7x16 transpose, both coordinates are transported
by fixed permutations of 112 states.

This module freezes those coordinate identities.  It does NOT choose which
local position a channel should store; that remains a task/learning rule.
"""

from __future__ import annotations

BLOCK_STATES = 112
CHANNELS = 16


def encode_h(row: int, col: int) -> tuple[int, int]:
    """112x112 pixel -> (H-phase block index, local 16x7 position)."""
    if not (0 <= row < 112 and 0 <= col < 112):
        raise ValueError("pixel outside 112x112 payload")

    block_row, local_row = divmod(row, 16)  # 7 blocks x 16 rows
    block_col, local_col = divmod(col, 7)   # 16 blocks x 7 cols

    block = 16 * block_row + block_col
    local = 7 * local_row + local_col
    return block, local


def decode_h(block: int, local: int) -> tuple[int, int]:
    """Inverse of encode_h."""
    if not (0 <= block < 112 and 0 <= local < 112):
        raise ValueError("block/local outside 112-state domain")

    block_row, block_col = divmod(block, 16)
    local_row, local_col = divmod(local, 7)

    row = 16 * block_row + local_row
    col = 7 * block_col + local_col
    return row, col


def encode_v(row: int, col: int) -> tuple[int, int]:
    """112x112 pixel -> (V-phase block index, local 7x16 position)."""
    if not (0 <= row < 112 and 0 <= col < 112):
        raise ValueError("pixel outside 112x112 payload")

    block_row, local_row = divmod(row, 7)   # 16 blocks x 7 rows
    block_col, local_col = divmod(col, 16)  # 7 blocks x 16 cols

    block = 7 * block_row + block_col
    local = 16 * local_row + local_col
    return block, local


def decode_v(block: int, local: int) -> tuple[int, int]:
    """Inverse of encode_v."""
    if not (0 <= block < 112 and 0 <= local < 112):
        raise ValueError("block/local outside 112-state domain")

    block_row, block_col = divmod(block, 7)
    local_row, local_col = divmod(local, 16)

    row = 7 * block_row + local_row
    col = 16 * block_col + local_col
    return row, col


def block_h_to_v(block: int) -> int:
    """Transport an H-phase 7x16 block-grid index into the V-phase 16x7 grid."""
    if not 0 <= block < 112:
        raise ValueError("block outside 112-state domain")
    block_row, block_col = divmod(block, 16)
    return 7 * block_col + block_row


def block_v_to_h(block: int) -> int:
    """Inverse block transport."""
    if not 0 <= block < 112:
        raise ValueError("block outside 112-state domain")
    block_row, block_col = divmod(block, 7)
    return 16 * block_col + block_row


def local_h_to_v(local: int) -> int:
    """Transpose one local 16x7 position into the local 7x16 indexing."""
    if not 0 <= local < 112:
        raise ValueError("local outside 112-state domain")
    local_row, local_col = divmod(local, 7)
    return 16 * local_col + local_row


def local_v_to_h(local: int) -> int:
    """Inverse local-position transpose."""
    if not 0 <= local < 112:
        raise ValueError("local outside 112-state domain")
    local_row, local_col = divmod(local, 16)
    return 7 * local_col + local_row


def transport_channel_positions_h_to_v(
    positions: list[list[int]],
) -> list[list[int]]:
    """Transport a 16x112 channel-position field without recomputation.

    positions[channel][H_block] = local H position (0..111)

    Returns:
        out[channel][V_block] = corresponding local V position.
    """
    if len(positions) != CHANNELS:
        raise ValueError("expected 16 channel rows")
    if any(len(row) != BLOCK_STATES for row in positions):
        raise ValueError("each channel row must contain 112 block positions")

    out = [[0] * BLOCK_STATES for _ in range(CHANNELS)]

    for channel in range(CHANNELS):
        for h_block in range(BLOCK_STATES):
            h_local = int(positions[channel][h_block])
            if not 0 <= h_local < BLOCK_STATES:
                raise ValueError("stored local position outside 0..111")

            v_block = block_h_to_v(h_block)
            v_local = local_h_to_v(h_local)
            out[channel][v_block] = v_local

    return out


def transport_channel_positions_v_to_h(
    positions: list[list[int]],
) -> list[list[int]]:
    """Exact inverse of transport_channel_positions_h_to_v."""
    if len(positions) != CHANNELS:
        raise ValueError("expected 16 channel rows")
    if any(len(row) != BLOCK_STATES for row in positions):
        raise ValueError("each channel row must contain 112 block positions")

    out = [[0] * BLOCK_STATES for _ in range(CHANNELS)]

    for channel in range(CHANNELS):
        for v_block in range(BLOCK_STATES):
            v_local = int(positions[channel][v_block])
            if not 0 <= v_local < BLOCK_STATES:
                raise ValueError("stored local position outside 0..111")

            h_block = block_v_to_h(v_block)
            h_local = local_v_to_h(v_local)
            out[channel][h_block] = h_local

    return out


def transport_block_polarity_h_to_v(polarity: list[int]) -> list[int]:
    """Transport one polarity metadata state per native block.

    Transpose changes position, not polarity, so only the block index moves.
    Values remain opaque uint8 states.
    """
    if len(polarity) != BLOCK_STATES:
        raise ValueError("expected 112 block-polarity states")

    out = [0] * BLOCK_STATES
    for h_block, value in enumerate(polarity):
        out[block_h_to_v(h_block)] = int(value) & 0xFF
    return out


def transport_block_polarity_v_to_h(polarity: list[int]) -> list[int]:
    """Inverse block-polarity transport."""
    if len(polarity) != BLOCK_STATES:
        raise ValueError("expected 112 block-polarity states")

    out = [0] * BLOCK_STATES
    for v_block, value in enumerate(polarity):
        out[block_v_to_h(v_block)] = int(value) & 0xFF
    return out
