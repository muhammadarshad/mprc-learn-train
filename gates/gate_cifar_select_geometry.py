#!/usr/bin/env python3
"""
CIFAR-10 real-data geometry audit for MPRC SELECT.

Purpose
-------
Test one narrow, label-blind hypothesis before training:

    A local 3x3 byte collision is ambiguous, but expanding the same candidate
    through MPRC observer support may make the query's true CIFAR class more
    recoverable.

Labels are NEVER used to construct K0, choose centers, choose thresholds, or
rank candidates.  They are read only after ranking for evaluation.

Native ingredients
------------------
- Z256 circular byte distance.
- Exact compact-support cloud numerator max(0, 64-d).
- Local 3x3 = 9 sites.
- Expanding Manhattan support S(r)=1+2r(r+1), r=1..7:
  5, 13, 25, 41, 61, 85, 113 sites.
- No resizing, interpolation, softmax, learned weights, or training.

The experiment deliberately searches local collisions across both image and
position. Candidate positions are all interior pixels that permit r=7.
"""

from __future__ import annotations

import json
import os
import pickle
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cifar10_raw"
ARCHIVE = CACHE / "cifar-10-python.tar.gz"
EXTRACTED = CACHE / "cifar-10-batches-py"
OUT = ROOT / "results" / "cifar_select_geometry.json"

RADII = tuple(range(1, 8))
SUPPORT_SIZES = {r: 1 + 2 * r * (r + 1) for r in RADII}
assert [SUPPORT_SIZES[r] for r in RADII] == [5, 13, 25, 41, 61, 85, 113]

# Fixed before labels are inspected.
N_QUERIES = 50
N_CANDIDATE_IMAGES = 200
K0 = 256
SEED = 20260927


def ensure_cifar() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if EXTRACTED.exists():
        return
    if not ARCHIVE.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        tf.extractall(CACHE)


def load_test() -> tuple[np.ndarray, np.ndarray, list[str]]:
    ensure_cifar()
    with open(EXTRACTED / "test_batch", "rb") as f:
        b = pickle.load(f, encoding="bytes")
    x = np.asarray(b[b"data"], dtype=np.uint8)
    y = np.asarray(b[b"labels"], dtype=np.int64)
    # CIFAR storage: R plane, G plane, B plane.
    x = x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    with open(EXTRACTED / "batches.meta", "rb") as f:
        m = pickle.load(f, encoding="bytes")
    names = [z.decode("utf-8") for z in m[b"label_names"]]
    return x, y, names


def ring_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return np.minimum(d, 256 - d)


def cloud_num_from_dist(d: np.ndarray) -> np.ndarray:
    # MPRC-CLOUD has factor 4; common factor is irrelevant to ranking.
    return np.maximum(0, 64 - d)


def diamond_offsets(r: int) -> np.ndarray:
    pts = []
    for dy in range(-r, r + 1):
        rem = r - abs(dy)
        for dx in range(-rem, rem + 1):
            pts.append((dy, dx))
    out = np.asarray(pts, dtype=np.int16)
    assert len(out) == SUPPORT_SIZES[r]
    return out


def square3_offsets() -> np.ndarray:
    return np.asarray([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=np.int16)


def gather_patch(img: np.ndarray, cy: int, cx: int, offsets: np.ndarray) -> np.ndarray:
    ys = cy + offsets[:, 0]
    xs = cx + offsets[:, 1]
    return img[ys, xs, :]


def score_candidate(
    qimg: np.ndarray,
    qcenter: tuple[int, int],
    cimg: np.ndarray,
    ccenter: tuple[int, int],
    offsets: np.ndarray,
) -> tuple[int, int]:
    q = gather_patch(qimg, qcenter[0], qcenter[1], offsets)
    c = gather_patch(cimg, ccenter[0], ccenter[1], offsets)
    d = ring_dist(q, c)
    # integer sums only; normalize only for report after ranking
    dist_sum = int(d.sum(dtype=np.int64))
    cloud_sum = int(cloud_num_from_dist(d).sum(dtype=np.int64))
    return dist_sum, cloud_sum


def main() -> None:
    x, y, names = load_test()
    n = len(x)
    assert n == 10000

    # Image choices and centers are index/seed driven only.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    qidx = perm[:N_QUERIES]
    cidx = perm[N_QUERIES:N_QUERIES + N_CANDIDATE_IMAGES]

    # All positions with complete radius-7 support.
    centers = np.asarray([(cy, cx) for cy in range(7, 25) for cx in range(7, 25)], dtype=np.int16)
    assert len(centers) == 324

    off9 = square3_offsets()
    off_r = {r: diamond_offsets(r) for r in RADII}

    # Build all candidate local 3x3 patches once.
    # Shape: candidate_images * 324, 9*3.
    local_blocks = []
    meta_img = []
    meta_center = []
    for ii in cidx:
        img = x[ii]
        for cy, cx in centers:
            local_blocks.append(gather_patch(img, int(cy), int(cx), off9).reshape(-1))
            meta_img.append(int(ii))
            meta_center.append((int(cy), int(cx)))
    cand_local = np.stack(local_blocks, axis=0).astype(np.uint8, copy=False)
    meta_img = np.asarray(meta_img, dtype=np.int32)
    meta_center = np.asarray(meta_center, dtype=np.int16)

    # Per-stage metrics.
    stages = ["local9"] + [f"r{r}_{SUPPORT_SIZES[r]}" for r in RADII]
    top1_correct = Counter()
    top5_same = Counter()
    top32_same = Counter()
    rank_same_sum = Counter()
    rank_same_found = Counter()
    n_eval = 0
    examples = []

    # Query centers: deterministic 7-stride-style walk through the 18x18 interior.
    for qi_ord, qi in enumerate(qidx):
        qcy = 7 + ((7 * qi_ord) % 18)
        qcx = 7 + ((13 * qi_ord) % 18)
        qpatch = gather_patch(x[qi], qcy, qcx, off9).reshape(1, -1)

        d = ring_dist(cand_local, qpatch)
        local_dist = d.sum(axis=1, dtype=np.int64)
        # K0: strongest local collisions, label blind.
        k0 = np.argpartition(local_dist, K0 - 1)[:K0]
        k0 = k0[np.argsort(local_dist[k0], kind="stable")]

        stage_orders: dict[str, np.ndarray] = {"local9": np.arange(K0, dtype=np.int32)}
        stage_scores: dict[str, list[int]] = {
            "local9": [int(local_dist[j]) for j in k0]
        }

        for r in RADII:
            vals = []
            for j in k0:
                ci = int(meta_img[j])
                cy, cx = (int(meta_center[j, 0]), int(meta_center[j, 1]))
                dist_sum, cloud_sum = score_candidate(x[qi], (qcy, qcx), x[ci], (cy, cx), off_r[r])
                vals.append((cloud_sum, -dist_sum))
            # higher cloud is better; lower distance breaks ties.
            order = np.asarray(sorted(range(K0), key=lambda z: vals[z], reverse=True), dtype=np.int32)
            stage = f"r{r}_{SUPPORT_SIZES[r]}"
            stage_orders[stage] = order
            stage_scores[stage] = [int(vals[z][0]) for z in order]

        qlabel = int(y[qi])
        row = {
            "query_index": int(qi),
            "query_label": qlabel,
            "query_class": names[qlabel],
            "query_center": [qcy, qcx],
            "stages": {},
        }

        for stage in stages:
            ord_local = stage_orders[stage]
            global_candidate_positions = k0[ord_local]
            labs = y[meta_img[global_candidate_positions]]
            same = (labs == qlabel)

            top1_correct[stage] += int(bool(same[0]))
            top5_same[stage] += int(same[:5].sum())
            top32_same[stage] += int(same[:32].sum())
            where = np.flatnonzero(same)
            if len(where):
                rank_same_sum[stage] += int(where[0] + 1)
                rank_same_found[stage] += 1

            row["stages"][stage] = {
                "top1_class": names[int(labs[0])],
                "top1_same": bool(same[0]),
                "top5_same_count": int(same[:5].sum()),
                "top32_same_count": int(same[:32].sum()),
                "best_same_rank": int(where[0] + 1) if len(where) else None,
            }
        if len(examples) < 12:
            examples.append(row)
        n_eval += 1

    metrics = {}
    for stage in stages:
        metrics[stage] = {
            "top1_same_class_rate": top1_correct[stage] / n_eval,
            "top5_same_class_fraction": top5_same[stage] / (n_eval * 5),
            "top32_same_class_fraction": top32_same[stage] / (n_eval * 32),
            "best_same_class_mean_rank_when_present": (
                rank_same_sum[stage] / rank_same_found[stage]
                if rank_same_found[stage] else None
            ),
            "queries_with_same_class_in_K0": int(rank_same_found[stage]),
        }

    result = {
        "experiment": "CIFAR-10 MPRC local-collision -> expanding-context class audit",
        "status": "COMPLETE",
        "dataset": {
            "source": URL,
            "split": "official test_batch",
            "images": int(n),
            "shape": [32, 32, 3],
            "classes": names,
        },
        "protocol": {
            "seed": SEED,
            "queries": N_QUERIES,
            "candidate_images": N_CANDIDATE_IMAGES,
            "candidate_centers_per_image": 324,
            "K0_local_collisions": K0,
            "query_center_rule": "[7+(7*i mod18), 7+(13*i mod18)]",
            "candidate_center_domain": "all [7..24]x[7..24]",
            "local": "3x3 RGB, Z256 circular byte distance",
            "expanded_measure": "sum max(0,64-cdist) over RGB diamond support; tie-break by lower cdist",
            "labels_used_for_ranking": False,
            "training": False,
            "softmax": False,
            "resizing": False,
            "support_sizes": [SUPPORT_SIZES[r] for r in RADII],
        },
        "query_label_distribution": {
            names[k]: int(v) for k, v in sorted(Counter(int(y[i]) for i in qidx).items())
        },
        "metrics": metrics,
        "examples": examples,
    }

    # Explicit comparison to local baseline.
    base = metrics["local9"]["top1_same_class_rate"]
    for stage in stages[1:]:
        result["metrics"][stage]["top1_delta_vs_local9"] = (
            result["metrics"][stage]["top1_same_class_rate"] - base
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
