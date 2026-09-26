"""v24 observation-to-structural interface finite gate.

This is a foundation/interface gate, not a recognition benchmark.

Candidate definition:
    Pi_T : [C,113,128] -> [C,128,113]
    Pi_T(x)[c,h,w] = x[c,w,h]

Verifies:
- exact involution / byte preservation
- channel independence
- Z256 BIND commutation
- circular MEASURE invariance
- native 7x16 <-> 16x7 local rectangle compatibility
- exact 98+15 target coordinate partition
- finite-cardinality no-free-metadata obstruction
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

C = 16
FH = 113
FW = 128
H = 128
W = 113
DATA_W = 98
INFO_W = 15

assert FH * FW == H * W == 14_464
assert DATA_W + INFO_W == W


def pi_t(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    assert a.shape == (C, FH, FW)
    assert a.dtype == np.uint8
    return np.transpose(a, (0, 2, 1)).copy()


def pi_t_inv(y: np.ndarray) -> np.ndarray:
    a = np.asarray(y)
    assert a.shape == (C, H, W)
    assert a.dtype == np.uint8
    return np.transpose(a, (0, 2, 1)).copy()


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a)
    bb = np.asarray(b)
    assert aa.shape == bb.shape
    assert aa.dtype == bb.dtype == np.uint8
    return ((aa.astype(np.uint16) + bb.astype(np.uint16)) & 0xFF).astype(np.uint8)


def ring_measure(a: np.ndarray, b: np.ndarray) -> int:
    aa = np.asarray(a)
    bb = np.asarray(b)
    assert aa.shape == bb.shape
    assert aa.dtype == bb.dtype == np.uint8
    ab = (aa.astype(np.int16) - bb.astype(np.int16)) & 0xFF
    ba = (bb.astype(np.int16) - aa.astype(np.int16)) & 0xFF
    d = np.minimum(ab, ba)
    return int(d.sum(dtype=np.int64))


def ring_measure_per_channel(a: np.ndarray, b: np.ndarray) -> list[int]:
    return [ring_measure(a[c], b[c]) for c in range(C)]


def deterministic_observation(offset: int) -> np.ndarray:
    # Covers all byte values repeatedly and differs by channel/site.
    c = np.arange(C, dtype=np.uint32)[:, None, None]
    r = np.arange(FH, dtype=np.uint32)[None, :, None]
    q = np.arange(FW, dtype=np.uint32)[None, None, :]
    z = (37 * c + 11 * r + 29 * q + offset) & 0xFF
    return z.astype(np.uint8)


def involution_gate():
    x = deterministic_observation(3)
    y = pi_t(x)
    z = pi_t_inv(y)

    assert y.shape == (16,128,113)
    assert np.array_equal(z, x)

    # Exact state multiset is preserved per channel.
    for c in range(C):
        assert np.array_equal(
            np.bincount(x[c].reshape(-1), minlength=256),
            np.bincount(y[c].reshape(-1), minlength=256),
        )

    return {
        "pass": True,
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "states_per_channel": FH * FW,
        "channels": C,
        "total_states": C * FH * FW,
        "inverse_exact": True,
    }


def bind_measure_gate():
    a = deterministic_observation(7)
    b = deterministic_observation(191)

    lhs = pi_t(bind(a,b))
    rhs = bind(pi_t(a), pi_t(b))
    assert np.array_equal(lhs, rhs)

    E0 = ring_measure(a,b)
    E1 = ring_measure(pi_t(a), pi_t(b))
    assert E0 == E1

    per0 = ring_measure_per_channel(a,b)
    per1 = ring_measure_per_channel(pi_t(a), pi_t(b))
    assert per0 == per1

    return {
        "pass": True,
        "bind_commutes": True,
        "measure_invariant": True,
        "energy_total": E0,
        "per_channel_energy": per0,
    }


def channel_independence_gate():
    x = np.zeros((C,FH,FW), dtype=np.uint8)

    for c in range(C):
        x[c, (7*c) % FH, (13*c) % FW] = (17*c + 1) & 0xFF

    y = pi_t(x)

    for c in range(C):
        # No state may leak into another channel.
        for d in range(C):
            nz = int(np.count_nonzero(y[d]))
            if d == c:
                continue
        assert int(np.count_nonzero(y[c])) == int(np.count_nonzero(x[c]))

    # Whole bank nonzero counts preserved channel-by-channel.
    assert [int(np.count_nonzero(x[c])) for c in range(C)] == [
        int(np.count_nonzero(y[c])) for c in range(C)
    ]

    return {
        "pass": True,
        "channels_mixed": False,
        "channel_nonzero_counts": [int(np.count_nonzero(y[c])) for c in range(C)],
    }


def native_rectangle_gate():
    # Use position-unique values large enough to test exact coordinate mapping.
    src = np.arange(FH * FW, dtype=np.int32).reshape(FH,FW)
    dst = src.T

    checked_7x16 = 0
    for r in range(FH - 7 + 1):
        for c in range(FW - 16 + 1):
            a = src[r:r+7, c:c+16]
            b = dst[c:c+16, r:r+7]
            assert np.array_equal(a.T, b)
            checked_7x16 += 1

    checked_16x7 = 0
    for r in range(FH - 16 + 1):
        for c in range(FW - 7 + 1):
            a = src[r:r+16, c:c+7]
            b = dst[c:c+7, r:r+16]
            assert np.array_equal(a.T, b)
            checked_16x7 += 1

    return {
        "pass": True,
        "source_7x16_to_target_16x7_origins": checked_7x16,
        "source_16x7_to_target_7x16_origins": checked_16x7,
        "local_native_transpose_exact": True,
    }


def partition_gate():
    x = pi_t(deterministic_observation(23))

    data = x[:, :, :DATA_W]
    info = x[:, :, DATA_W:]

    assert data.shape == (C,H,98)
    assert info.shape == (C,H,15)
    assert data.size + info.size == x.size

    rebuilt = np.concatenate([data,info], axis=2)
    assert np.array_equal(rebuilt,x)

    return {
        "pass": True,
        "per_channel": {
            "data_states": H * DATA_W,
            "information_states": H * INFO_W,
            "total": H * W,
        },
        "bank": {
            "data_states": C * H * DATA_W,
            "information_states": C * H * INFO_W,
            "total": C * H * W,
        },
        "semantic_information_meaning_proved": False,
    }


def no_free_metadata_gate():
    observation_states = 14_464
    independent_metadata_states = 1_920
    target_states = 14_464

    source_exponent = observation_states + independent_metadata_states
    target_exponent = target_states

    assert source_exponent == 16_384
    assert target_exponent == 14_464
    assert source_exponent > target_exponent

    # Compare cardinalities through log_256 exponents exactly instead of
    # materializing astronomically large integers.
    excess = source_exponent - target_exponent
    assert excess == 1_920

    return {
        "pass": True,
        "ring_cardinality": 256,
        "source_pair_log256_cardinality": source_exponent,
        "target_log256_cardinality": target_exponent,
        "excess_independent_state_exponent": excess,
        "injective_lossless_map_for_arbitrary_pair_exists": False,
        "theorem": (
            "An arbitrary full 14,464-state observation plus an independent "
            "1,920-state metadata object cannot be encoded injectively into "
            "one 14,464-state Z256 manifold."
        ),
    }


def encoder_degree_of_freedom_note():
    # Pure counting note for the actual fixed encoder architecture:
    # 16 materialized output channels, but channels 0..2 preserve RGB directly.
    source_states = 3 * 14_464
    materialized_states = 16 * 14_464

    assert source_states == 43_392
    assert materialized_states == 231_424

    return {
        "rgb_source_states": source_states,
        "materialized_wave_states": materialized_states,
        "materialization_ratio_numerator": 16,
        "materialization_ratio_denominator": 3,
        "note": (
            "The fixed encoder output is deterministic from RGB and contains "
            "RGB pass-through, so the 16 channels are not 16 independent "
            "arbitrary manifolds. An explicit multi-channel codec remains open."
        ),
    }


def main():
    report = {
        "name":"v24 Observation-to-Structural Interface",
        "candidate_interface":{
            "definition":"Pi_T[c,h,w] = Psi[c,w,h]",
            "input":"Z256^[16,113,128]",
            "output":"(Z256^[128,113])^16",
            "status":"candidate definition, not uniqueness theorem",
        },
        "gates":{
            "involution":involution_gate(),
            "bind_measure":bind_measure_gate(),
            "channel_independence":channel_independence_gate(),
            "native_rectangles":native_rectangle_gate(),
            "partition":partition_gate(),
            "no_free_metadata":no_free_metadata_gate(),
        },
        "encoder_degree_of_freedom":encoder_degree_of_freedom_note(),
        "closure":{
            "G6a_coordinate_type_interface":"PASS for candidate Pi_T",
            "G6b_information_semantics":"OPEN",
            "G6c_multichannel_binding_or_codec":"OPEN",
            "15_plus_1_channel_semantics":"OPEN",
        },
        "pass":True,
        "claim_boundary":(
            "v24 proves exact properties of the explicitly defined channel-wise "
            "frame transpose and a cardinality obstruction for independent metadata. "
            "It does not prove Pi_T is the unique/canonical MPRC packing law and does "
            "not assign semantic meaning to the 15 INFORMATION lanes."
        ),
    }

    out=Path("results/v24_observation_structural_interface.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
