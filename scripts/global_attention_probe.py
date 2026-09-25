"""Global Attention probe for MPRC local/global Arshad Transpose.

NO CLASSIFIER. NO LABELS. NO SOFTMAX. NO DENSE ATTENTION MATRIX.

Pipeline under probe:

    query occurrence
      -> IDENTIFY local p in Z256
      -> local relation policy
      -> Arshad Transpose X^T[p]
      -> globally unique candidates g in 14,464*N
      -> BIND -> REACT -> MEASURE diagnostic

The probe uses controlled synthetic manifolds so retrieval truth is exact.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from mprc_structural.global_attention import (
    D,
    HV,
    ExactStatePolicy,
    GeneratorPrefixPolicy,
    TransposeIndex,
    global_address,
    retrieve,
    measure_candidate_manifolds,
)
from mprc_structural.manifold import H, W, DATA_W, INFO_W, GEN

QUERY_P = 77
PREFIXES = (1, 2, 4, 7, 14, 16, 32, 64)
PROBE_N = 8


def generator_prefix(p: int, k: int) -> list[int]:
    return [((p + GEN * t) & 0xFF) for t in range(k)]


def build_controlled_manifolds(N: int) -> tuple[np.ndarray, dict]:
    """Create exact retrieval truth.

    Each sample contains one occurrence of each of the first 64 generator states,
    placed in DATA lanes 0..63 of one row. INFORMATION lanes are sample-specific
    values chosen outside those 64 states.
    """
    rel64 = set(generator_prefix(QUERY_P, 64))
    complement = [p for p in range(256) if p not in rel64]
    assert len(complement) == 192

    background = complement[0]
    manifolds = np.full((N, H, W), background, dtype=np.uint8)

    # Distinct INFORMATION context per sample, all outside relation prefix states.
    info_values = []
    for n in range(N):
        v = complement[n + 1]
        info_values.append(v)
        manifolds[n, :, DATA_W:] = v

    row = 20
    planted: dict[int, list[int]] = {}
    for t, p in enumerate(generator_prefix(QUERY_P, 64)):
        col = t
        assert col < DATA_W
        h = row * W + col
        planted[p] = []
        for n in range(N):
            manifolds[n, row, col] = p
            planted[p].append(global_address(n, h, N))

    target_h = row * W
    truth = {
        "query_local_state": QUERY_P,
        "target_site": target_h,
        "background": background,
        "information_values": info_values,
        "planted_globals_by_state": planted,
    }
    return manifolds, truth


def count_expected_planted(truth: dict, states: list[int]) -> set[int]:
    out: set[int] = set()
    planted = truth["planted_globals_by_state"]
    for p in states:
        for g in planted.get(int(p), []):
            out.add(int(g))
    return out


report = {
    "local_domain": D,
    "hv": HV,
    "global_domain": "14464*N",
    "generator": GEN,
}

# ------------------------------------------------------------------
# A. Controlled local/global transpose
# ------------------------------------------------------------------
manifolds, truth = build_controlled_manifolds(PROBE_N)
flat = manifolds.reshape(-1)
index = TransposeIndex.from_manifolds(manifolds)
reconstructed = index.reconstruct()

assert np.array_equal(reconstructed, flat)
assert len(index.offsets) == 257
assert len(index.positions) == HV * PROBE_N

counts = np.diff(index.offsets)

report["transpose"] = {
    "N": PROBE_N,
    "roundtrip": True,
    "positions_preserved": f"{len(index.positions)}/{HV * PROBE_N}",
    "bucket_min": int(counts.min()),
    "bucket_max": int(counts.max()),
    "offset_count": int(len(index.offsets)),
    "position_count": int(len(index.positions)),
    "dense_matrix_required": False,
}

# ------------------------------------------------------------------
# B. Exact-state self retrieval
# ------------------------------------------------------------------
exact = retrieve(index, QUERY_P, ExactStatePolicy())
exact_globals = {c.global_position for c in exact}

self_ok = 0
for n in range(PROBE_N):
    g = global_address(n, truth["target_site"], PROBE_N)
    if g in exact_globals:
        self_ok += 1

assert self_ok == PROBE_N

report["self_retrieval"] = {
    "numerator": self_ok,
    "denominator": PROBE_N,
    "exact_state_candidate_count": len(exact),
}

# ------------------------------------------------------------------
# C. Duplicate-local global identity
# ------------------------------------------------------------------
target_globals = set(truth["planted_globals_by_state"][QUERY_P])
assert target_globals.issubset(exact_globals)
assert len(target_globals) == PROBE_N

report["duplicate_local"] = {
    "local_state": QUERY_P,
    "expected_global_occurrences": PROBE_N,
    "preserved_global_occurrences": len(target_globals),
    "expected_globals": sorted(target_globals),
}

# ------------------------------------------------------------------
# D. Same local p, different global INFORMATION context -> MEASURE diagnostic
# ------------------------------------------------------------------
query_manifold = manifolds[0]
measure_rows = measure_candidate_manifolds(
    query_manifold,
    manifolds,
    [c for c in exact if c.global_position in target_globals],
    rounds=1,
)

# One target occurrence per sample.
target_measure = []
seen_samples = set()
for row in measure_rows:
    if row["sample"] in seen_samples:
        continue
    seen_samples.add(row["sample"])
    target_measure.append(row)

energies = [r["energy"] for r in target_measure]
distinct_energies = len(set(energies))

report["measure"] = {
    "same_local_state": QUERY_P,
    "candidate_samples": len(target_measure),
    "distinct_energy_count": distinct_energies,
    "globally_distinguished": distinct_energies > 1,
    "ordering": target_measure,
    "note": (
        "Diagnostic only. Same local state is preserved as distinct global candidates; "
        "whether current BIND/REACT/MEASURE distinguishes their INFORMATION context is empirical."
    ),
}

# ------------------------------------------------------------------
# E. Generator relation prefixes
# ------------------------------------------------------------------
prefix_report = {}
for K in PREFIXES:
    policy = GeneratorPrefixPolicy(K)
    local_states = policy.states(QUERY_P)
    candidates = retrieve(index, QUERY_P, policy)

    got_globals = {c.global_position for c in candidates}
    expected_planted = count_expected_planted(truth, local_states)
    recalled = len(got_globals & expected_planted)

    assert recalled == len(expected_planted)

    # Measure ordering on globally retrieved candidates. Energies are computed once
    # per unique candidate manifold by the reference helper.
    measured = measure_candidate_manifolds(
        query_manifold,
        manifolds,
        candidates,
        rounds=1,
    )

    # Compact sample-level ordering so JSON does not explode for broad prefixes.
    first_per_sample = {}
    for r in measured:
        s = r["sample"]
        if s not in first_per_sample:
            first_per_sample[s] = r

    prefix_report[str(K)] = {
        "local_states": local_states,
        "local_state_count": len(local_states),
        "global_candidates": len(candidates),
        "unique_global_candidates": len(got_globals),
        "expected_planted_recall": {
            "numerator": recalled,
            "denominator": len(expected_planted),
        },
        "duplicates_after_union": len(candidates) - len(got_globals),
        "sample_measure_ordering": sorted(
            first_per_sample.values(),
            key=lambda r: (r["energy"], r["sample"]),
        ),
    }

report["generator_prefix"] = prefix_report

# ------------------------------------------------------------------
# F. Scaling of sparse transpose
# ------------------------------------------------------------------
scaling = {}
for N in (1, 2, 7, 16, 64):
    L = HV * N
    # Deterministic full byte-domain witness; no labels.
    states = (np.arange(L, dtype=np.uint32) & 0xFF).astype(np.uint8)

    t0 = time.perf_counter_ns()
    idx = TransposeIndex.build(states, N)
    t1 = time.perf_counter_ns()

    # Query a bucket repeatedly for stable lookup accounting.
    q0 = time.perf_counter_ns()
    total = 0
    for _ in range(10_000):
        total += len(idx.bucket(QUERY_P))
    q1 = time.perf_counter_ns()
    assert total > 0

    assert np.array_equal(idx.reconstruct(), states)

    sparse_bytes = int(idx.offsets.nbytes + idx.positions.nbytes)
    dense_byte_membership = D * L  # one byte per local/global membership cell
    avoided = dense_byte_membership - sparse_bytes

    c = np.diff(idx.offsets)

    scaling[str(N)] = {
        "global_states": L,
        "build_time_ns": int(t1 - t0),
        "offsets_bytes": int(idx.offsets.nbytes),
        "positions_bytes": int(idx.positions.nbytes),
        "sparse_index_bytes": sparse_bytes,
        "bucket_min": int(c.min()),
        "bucket_max": int(c.max()),
        "query_10000_time_ns": int(q1 - q0),
        "query_bucket_size": int(len(idx.bucket(QUERY_P))),
        "dense_byte_membership_equivalent": int(dense_byte_membership),
        "dense_equivalent_bytes_avoided": int(avoided),
        "dense_assumption": "one byte per cell for hypothetical 256 x (HV*N) membership table",
    }

report["scaling"] = scaling

report["overall"] = "PROBE COMPLETE"
report["training_gate_open"] = False
report["claim_boundary"] = (
    "Transpose retrieval, self inclusion, global identity preservation and sparse scaling "
    "are exact properties. Generator-prefix relevance and BIND/REACT/MEASURE ordering are "
    "probe diagnostics, not learned-attention or recognition claims."
)

root = Path(__file__).resolve().parents[1]
out = root / "results" / "global_attention_probe.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
