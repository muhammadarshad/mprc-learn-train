import numpy as np

from mprc_structural.global_attention import (
    D,
    HV,
    ExactStatePolicy,
    ExplicitRelationPolicy,
    GeneratorPrefixPolicy,
    TransposeIndex,
    decode_global,
    global_address,
    retrieve,
)
from mprc_structural.manifold import H, W


def test_global_address_roundtrip():
    for N in (1, 2, 7):
        for n in range(N):
            for h in (0, 1, HV // 2, HV - 1):
                g = global_address(n, h, N)
                assert decode_global(g, N) == (n, h)


def test_transpose_roundtrip_random():
    rng = np.random.default_rng(20260925)
    for N in (1, 2, 7):
        x = rng.integers(0, 256, size=HV * N, dtype=np.uint8)
        idx = TransposeIndex.build(x, N)
        assert np.array_equal(idx.reconstruct(), x)


def test_duplicate_local_state_preserves_global_occurrences():
    N = 3
    x = np.zeros(HV * N, dtype=np.uint8)
    targets = [0, 1, HV - 1, HV, HV + 1, 2 * HV, 3 * HV - 1]
    for g in targets:
        x[g] = 77
    idx = TransposeIndex.build(x, N)
    assert idx.bucket(77).tolist() == targets


def test_exact_policy_self_retrieval():
    N = 2
    manifolds = np.zeros((N, H, W), dtype=np.uint8)
    h = 31 * W + 97
    manifolds[:, 31, 97] = 77

    idx = TransposeIndex.from_manifolds(manifolds)
    candidates = retrieve(idx, 77, ExactStatePolicy())
    globals_ = {c.global_position for c in candidates}

    for n in range(N):
        assert global_address(n, h, N) in globals_


def test_generator_prefix_is_generator7():
    states = GeneratorPrefixPolicy(7).states(10)
    assert states == [(10 + 7 * t) & 255 for t in range(7)]
    assert len(set(states)) == 7


def test_explicit_policy():
    p = ExplicitRelationPolicy((3, 9, 255))
    assert p.states(200) == [3, 9, 255]


def test_collect_deduplicates_global_identity_only():
    N = 1
    x = np.arange(HV, dtype=np.int64) & 255
    idx = TransposeIndex.build(x.astype(np.uint8), N)

    # Same local relation listed twice must not duplicate its global occurrences.
    a = idx.collect([7, 7])
    b = idx.collect([7])
    assert a == b
