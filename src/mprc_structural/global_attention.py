"""Global Attention primitives for MPRC local/global state retrieval.

LOCAL:
    p in Z256

GLOBAL:
    one manifold has HV = 128*113 = 14,464 positions
    N manifolds have HV*N globally unique occurrences

    g = n*HV + h

Arshad Transpose:
    X : G_N -> Z256
    X^T[p] = {g : X(g)=p}

The transpose is stored sparsely as CSR-like offsets[257] + positions[HV*N].
No dense 256 x (HV*N) matrix is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from .manifold import H, W, GEN
from .attention import attention_forward

D = 256
HV = H * W
GEN_INV = pow(GEN, -1, D)

assert HV == 14_464
assert GEN == 7
assert GEN_INV == 183


def global_address(sample: int, site: int, n_samples: int) -> int:
    n = int(sample)
    h = int(site)
    N = int(n_samples)
    if not 0 <= n < N:
        raise ValueError("sample outside global domain")
    if not 0 <= h < HV:
        raise ValueError("site outside manifold")
    return n * HV + h


def decode_global(g: int, n_samples: int) -> tuple[int, int]:
    x = int(g)
    N = int(n_samples)
    if not 0 <= x < HV * N:
        raise ValueError("global address outside domain")
    return divmod(x, HV)


@dataclass(frozen=True)
class Candidate:
    local_state: int
    global_position: int
    sample: int
    site: int


@dataclass
class TransposeIndex:
    """Sparse inverted index from local Z256 state to global occurrence IDs."""

    offsets: np.ndarray
    positions: np.ndarray
    n_samples: int

    @classmethod
    def build(cls, states: np.ndarray, n_samples: int) -> "TransposeIndex":
        x = np.asarray(states)
        N = int(n_samples)
        expected = HV * N
        if x.ndim != 1 or x.shape[0] != expected:
            raise ValueError(f"expected flat state vector of length {expected}")
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError("states must be integer bytes")
        if np.any((x < 0) | (x > 255)):
            raise ValueError("state outside Z256")

        xb = x.astype(np.uint8, copy=False)
        counts = np.bincount(xb, minlength=D).astype(np.int64)
        offsets = np.empty(D + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])

        # Stable sort preserves ascending global occurrence order within bucket.
        positions = np.argsort(xb, kind="stable").astype(np.int64)
        assert int(offsets[-1]) == expected
        return cls(offsets=offsets, positions=positions, n_samples=N)

    @classmethod
    def from_manifolds(cls, manifolds: np.ndarray) -> "TransposeIndex":
        x = np.asarray(manifolds)
        if x.ndim != 3 or x.shape[1:] != (H, W):
            raise ValueError(f"expected [N,{H},{W}] uint8 manifolds")
        if x.dtype != np.uint8:
            raise TypeError("manifolds must be uint8")
        return cls.build(x.reshape(-1), x.shape[0])

    @property
    def global_size(self) -> int:
        return HV * self.n_samples

    def bucket(self, p: int) -> np.ndarray:
        q = int(p)
        if not 0 <= q < D:
            raise ValueError("local state outside Z256")
        a = int(self.offsets[q])
        b = int(self.offsets[q + 1])
        return self.positions[a:b]

    def candidates(self, p: int) -> list[Candidate]:
        q = int(p)
        out: list[Candidate] = []
        for g0 in self.bucket(q):
            g = int(g0)
            n, h = decode_global(g, self.n_samples)
            out.append(Candidate(q, g, n, h))
        return out

    def collect(self, local_states: Iterable[int]) -> list[Candidate]:
        """Union candidate occurrences, preserving global identity."""
        seen: set[int] = set()
        out: list[Candidate] = []
        for p0 in local_states:
            p = int(p0) & 0xFF
            for c in self.candidates(p):
                if c.global_position in seen:
                    continue
                seen.add(c.global_position)
                out.append(c)
        out.sort(key=lambda c: c.global_position)
        return out

    def reconstruct(self) -> np.ndarray:
        out = np.empty(self.global_size, dtype=np.uint8)
        filled = np.zeros(self.global_size, dtype=np.bool_)
        for p in range(D):
            b = self.bucket(p)
            if np.any(filled[b]):
                raise AssertionError("global occurrence appears in multiple buckets")
            out[b] = p
            filled[b] = True
        if not bool(filled.all()):
            raise AssertionError("global occurrence missing from transpose")
        return out


class LocalRelationPolicy(Protocol):
    def states(self, query_local_state: int) -> list[int]:
        ...


@dataclass(frozen=True)
class ExactStatePolicy:
    def states(self, query_local_state: int) -> list[int]:
        return [int(query_local_state) & 0xFF]


@dataclass(frozen=True)
class GeneratorPrefixPolicy:
    k: int

    def states(self, query_local_state: int) -> list[int]:
        K = int(self.k)
        if not 1 <= K <= D:
            raise ValueError("k must be 1..256")
        q = int(query_local_state) & 0xFF
        return [((q + GEN * t) & 0xFF) for t in range(K)]


@dataclass(frozen=True)
class ExplicitRelationPolicy:
    related_states: tuple[int, ...]

    def states(self, query_local_state: int) -> list[int]:
        del query_local_state
        return [int(p) & 0xFF for p in self.related_states]


def retrieve(
    index: TransposeIndex,
    query_local_state: int,
    policy: LocalRelationPolicy,
) -> list[Candidate]:
    return index.collect(policy.states(query_local_state))


def measure_candidate_manifolds(
    query_manifold: np.ndarray,
    manifolds: np.ndarray,
    candidates: list[Candidate],
    *,
    rounds: int = 1,
    lut: np.ndarray | None = None,
) -> list[dict]:
    """Run BIND->REACT->MEASURE on unique candidate manifolds.

    Candidate global positions remain attached to the result. Multiple candidate
    occurrences in the same sample share one manifold energy but are not collapsed
    from the retrieval result itself.
    """
    q = np.asarray(query_manifold)
    ms = np.asarray(manifolds)
    if q.shape != (H, W) or q.dtype != np.uint8:
        raise ValueError("query_manifold must be uint8 [128,113]")
    if ms.ndim != 3 or ms.shape[1:] != (H, W) or ms.dtype != np.uint8:
        raise ValueError("manifolds must be uint8 [N,128,113]")

    by_sample: dict[int, int] = {}
    for c in candidates:
        if c.sample not in by_sample:
            out = attention_forward(
                ms[c.sample],
                q,
                lut=lut,
                rounds=rounds,
                return_physical_state=False,
            )
            by_sample[c.sample] = int(out["energy"])

    rows = [
        {
            "local_state": c.local_state,
            "global_position": c.global_position,
            "sample": c.sample,
            "site": c.site,
            "energy": by_sample[c.sample],
        }
        for c in candidates
    ]
    rows.sort(key=lambda r: (r["energy"], r["global_position"]))
    return rows
