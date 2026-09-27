"""Typed directional IDENTIFY transpose and instrumented ArshadBlock.

Candidate rule frozen before labels/datasets:
    I(g) in Z256^9 is the canonical directional ADI-9 descriptor.
    I^T[D] = {g : I(g) = D}.

This is an exact sparse inverse image of IDENTIFY.  It does not use Top-K,
radius, learned similarity, 512-state packing, or a dense attention matrix.

H/V orientation is canonicalized by the proved directional ADI transpose.
QH4 tags are lossless type annotations of each descriptor byte.

The block then executes, in order:
    IDENTIFY -> exact typed transpose -> generator-7 TRANSPORT
    -> BIND -> REACT -> MEASURE.

This file defines a candidate routing rule.  Its usefulness is empirical;
its exactness/participation must pass the no-dataset survival gate first.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np

from .manifold import H, W
from .directional_adi import encode as adi_encode, transpose as adi_transpose
from .identify_qh4_adi import qh4_tag, qh4_untag
from .attention import (
    transport_manifold, bind, react, measure_energy,
)

HV = H * W


def _descriptor(values9, phase: str = "H") -> tuple[int, ...]:
    """Raw directional bytes -> canonical H-phase ADI-9 descriptor."""
    z = tuple(int(v) & 0xFF for v in adi_encode(values9))
    ph = str(phase).upper()
    if ph == "H":
        return z
    if ph == "V":
        return tuple(int(v) & 0xFF for v in adi_transpose(z))
    raise ValueError("phase must be H or V")


def _validate_qh4_descriptor(desc: tuple[int, ...]) -> tuple:
    """Type every descriptor byte and prove exact untagging."""
    if len(desc) != 9:
        raise ValueError("descriptor must have 9 bytes")
    tags = tuple(qh4_tag(v) for v in desc)
    rebuilt = tuple(qh4_untag(t) for t in tags)
    if rebuilt != desc:
        raise AssertionError("QH4 tag round-trip failed")
    return tags


@dataclass(frozen=True)
class TypedCandidate:
    descriptor: tuple[int, ...]
    global_position: int
    sample: int
    site: int
    anchor: int


@dataclass
class TypedDescriptorIndex:
    """Sparse exact transpose of the typed 9-byte IDENTIFY map."""

    buckets: dict[bytes, np.ndarray]
    sites: np.ndarray
    n_samples: int
    canonical_descriptors: np.ndarray

    @classmethod
    def build(
        cls,
        raw_values: np.ndarray,
        sites: np.ndarray,
        phases: np.ndarray | None = None,
    ) -> "TypedDescriptorIndex":
        x = np.asarray(raw_values)
        if x.ndim != 3 or x.shape[2] != 9:
            raise ValueError("raw_values must be [N,A,9]")
        if x.dtype != np.uint8:
            raise TypeError("raw_values must be uint8")

        N, A, _ = x.shape
        st = np.asarray(sites, dtype=np.int64)
        if st.shape != (A,):
            raise ValueError("sites must have one manifold site per anchor")
        if np.any((st < 0) | (st >= HV)):
            raise ValueError("site outside 128x113 manifold")
        if len(set(map(int, st))) != A:
            raise ValueError("anchor sites must be unique")

        if phases is None:
            ph = np.full((N, A), "H", dtype="<U1")
        else:
            ph = np.asarray(phases)
            if ph.shape != (N, A):
                raise ValueError("phases must be [N,A]")

        canon = np.empty((N, A, 9), dtype=np.uint8)
        tmp: dict[bytes, list[int]] = {}

        for n in range(N):
            for a in range(A):
                d = _descriptor(x[n, a], str(ph[n, a]))
                _validate_qh4_descriptor(d)
                canon[n, a] = np.asarray(d, dtype=np.uint8)
                g = n * HV + int(st[a])
                tmp.setdefault(bytes(d), []).append(g)

        buckets = {
            k: np.asarray(sorted(v), dtype=np.int64)
            for k, v in tmp.items()
        }
        return cls(
            buckets=buckets,
            sites=st.copy(),
            n_samples=N,
            canonical_descriptors=canon,
        )

    def bucket(self, descriptor: tuple[int, ...]) -> np.ndarray:
        d = tuple(int(v) & 0xFF for v in descriptor)
        if len(d) != 9:
            raise ValueError("descriptor must have 9 bytes")
        return self.buckets.get(bytes(d), np.empty(0, dtype=np.int64))

    def candidates(self, descriptor: tuple[int, ...]) -> list[TypedCandidate]:
        d = tuple(int(v) & 0xFF for v in descriptor)
        site_to_anchor = {int(s): i for i, s in enumerate(self.sites)}
        out = []
        for g0 in self.bucket(d):
            g = int(g0)
            n, site = divmod(g, HV)
            anchor = site_to_anchor.get(site)
            if anchor is None:
                raise AssertionError("indexed site has no anchor")
            out.append(TypedCandidate(d, g, n, site, anchor))
        return out

    def reconstruct(self) -> np.ndarray:
        """Reconstruct the canonical descriptor field exactly from sparse buckets."""
        N = self.n_samples
        A = len(self.sites)
        site_to_anchor = {int(s): i for i, s in enumerate(self.sites)}
        out = np.empty((N, A, 9), dtype=np.uint8)
        filled = np.zeros((N, A), dtype=np.bool_)

        for key, gs in self.buckets.items():
            d = np.frombuffer(key, dtype=np.uint8)
            if d.shape != (9,):
                raise AssertionError("bad descriptor key")
            for g0 in gs:
                n, site = divmod(int(g0), HV)
                a = site_to_anchor.get(site)
                if a is None or filled[n, a]:
                    raise AssertionError("duplicate or invalid occurrence")
                out[n, a] = d
                filled[n, a] = True

        if not bool(filled.all()):
            raise AssertionError("descriptor occurrence lost")
        return out


def _digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


class ArshadBlock:
    """Instrumented candidate block; no labels and no trainable routing."""

    def __init__(self, lut: np.ndarray | None = None, rounds: int = 7):
        self.lut = None if lut is None else np.asarray(lut, dtype=np.uint8)
        if self.lut is not None and self.lut.shape != (256,):
            raise ValueError("lut must have 256 entries")
        self.rounds = int(rounds)
        if self.rounds < 1:
            raise ValueError("rounds must be >=1")

    def forward(
        self,
        query_values9,
        query_manifold: np.ndarray,
        candidate_manifolds: np.ndarray,
        index: TypedDescriptorIndex,
        *,
        phase: str = "H",
    ) -> dict:
        q = np.asarray(query_manifold)
        ms = np.asarray(candidate_manifolds)
        if q.shape != (H, W) or q.dtype != np.uint8:
            raise ValueError("query_manifold must be uint8 [128,113]")
        if ms.ndim != 3 or ms.shape[1:] != (H, W) or ms.dtype != np.uint8:
            raise ValueError("candidate_manifolds must be uint8 [N,128,113]")
        if ms.shape[0] != index.n_samples:
            raise ValueError("index/manifold sample mismatch")

        # IDENTIFY
        desc = _descriptor(query_values9, phase)
        tags = _validate_qh4_descriptor(desc)

        # Exact typed transpose (candidate action of IDENTIFY).
        candidates = index.candidates(desc)

        # TRANSPORT query once.
        tq = transport_manifold(q)

        sample_cache: dict[int, dict] = {}
        rows = []

        for c in candidates:
            if c.sample not in sample_cache:
                physical = ms[c.sample]
                transported = transport_manifold(physical)
                bound = bind(transported, tq)
                reacted = react(bound, lut=self.lut, rounds=self.rounds)
                energy = measure_energy(reacted, tq)

                sample_cache[c.sample] = {
                    "energy": int(energy),
                    "trace": {
                        "executed": [
                            "IDENTIFY_ADI9_QH4",
                            "TYPED_TRANSPOSE",
                            "GENERATOR7_TRANSPORT",
                            "BIND",
                            f"REACT_x{self.rounds}",
                            "MEASURE",
                        ],
                        "physical_sha256": _digest(physical),
                        "transported_sha256": _digest(transported),
                        "query_transport_sha256": _digest(tq),
                        "bound_sha256": _digest(bound),
                        "reacted_sha256": _digest(reacted),
                    },
                }

            rows.append({
                "global_position": c.global_position,
                "sample": c.sample,
                "site": c.site,
                "anchor": c.anchor,
                "energy": sample_cache[c.sample]["energy"],
            })

        rows.sort(key=lambda r: (r["energy"], r["global_position"]))
        return {
            "descriptor": desc,
            "qh4_tags": tags,
            "candidate_count": len(candidates),
            "candidates": rows,
            "sample_traces": sample_cache,
            "rounds": self.rounds,
        }
