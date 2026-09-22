"""
Gap-closure empirical benchmark for MPRC observation-populated learning.

Frozen/recovered ingredients:
- Z256 / QH4 quarter structure.
- generator-7 edge transport schedule.
- deterministic u8 vision channels recovered from prior Arshad-ViT experiments.
- observation-populated categorical LUT; no gradient update and no Softmax.

This script intentionally keeps the learned architecture fixed:
    channels = gy, grad, h2, m4
    QH4 relation state = quarter pair
    smoothing alpha = 0.05
    integer score scale = 1

It reports:
- five fixed stratified train/test splits;
- same-information GD controls;
- random-label and per-image pixel-shuffle controls;
- translation and 90-degree rotation robustness.

The result is empirical evidence, not a theorem.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = np.load(ROOT / "data" / "digits_seed42.npz")
X = DATA["X"].astype(np.uint8).reshape(-1, 8, 8)
y = DATA["y"].astype(int)

K = 10
ALPHA = 0.05
SCORE_SCALE = 1
SELECTED = ["gy", "grad", "h2", "m4"]
SPLIT_SEEDS = [7, 19, 42, 71, 113]

U8 = ((X.astype(np.uint16) * 255) // 16).astype(np.uint8)


def blur(a, r):
    return np.array(
        Image.fromarray(a, "L").filter(ImageFilter.BoxBlur(r)),
        dtype=np.uint8,
    )


def adiff(a, b):
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)


def channels(img):
    luma = img.astype(np.uint8)

    gx = np.zeros_like(luma, dtype=np.int16)
    gy = np.zeros_like(luma, dtype=np.int16)
    gx[:, 1:-1] = luma[:, 2:].astype(np.int16) - luma[:, :-2].astype(np.int16)
    gy[1:-1, :] = luma[2:, :].astype(np.int16) - luma[:-2, :].astype(np.int16)

    gy_u8 = np.clip(128 + gy // 2, 0, 255).astype(np.uint8)
    grad = np.clip((np.abs(gx) + np.abs(gy)) // 2, 0, 255).astype(np.uint8)

    b1, b2, b4, b8 = [blur(luma, r) for r in (1, 2, 4, 8)]

    return {
        "gy": gy_u8,
        "grad": grad,
        "h2": adiff(b1, b2),
        "m4": adiff(b2, b4),
    }


CHANNELS = [channels(im) for im in U8]

# QH4/ViT relation topology: horizontal, vertical, both diagonals, generator-7 orbit.
orbit = []
q = 0
for _ in range(64):
    orbit.append(q)
    q = (q + 7) & 63
assert len(set(orbit)) == 64

EDGES = []
for r in range(8):
    for c in range(7):
        EDGES.append((r * 8 + c, r * 8 + c + 1))
for r in range(7):
    for c in range(8):
        EDGES.append((r * 8 + c, (r + 1) * 8 + c))
for r in range(7):
    for c in range(7):
        EDGES.append((r * 8 + c, (r + 1) * 8 + c + 1))
        EDGES.append((r * 8 + c + 1, (r + 1) * 8 + c))
for i in range(64):
    EDGES.append((orbit[i], orbit[(i + 1) % 64]))
assert len(EDGES) == 274


def qpair(img):
    f = img.reshape(-1)
    return np.asarray(
        [((int(f[i]) >> 6) << 2) | (int(f[j]) >> 6) for i, j in EDGES],
        dtype=np.uint8,
    )


Q = {
    name: np.stack([qpair(c[name]) for c in CHANNELS])
    for name in SELECTED
}

GQ = np.concatenate([Q[name] for name in SELECTED], axis=1).astype(np.int16)
GRAW = np.concatenate(
    [
        np.stack([c[name] for c in CHANNELS]).reshape(len(CHANNELS), -1)
        for name in SELECTED
    ],
    axis=1,
).astype(np.float64)


def train_integer_lut(train_idx, labels):
    tabs = {}
    for name in SELECTED:
        A = Q[name][train_idx]
        counts = np.zeros((A.shape[1], K, 16), dtype=np.int32)

        for cls in range(K):
            rows = A[labels == cls]
            for j in range(A.shape[1]):
                counts[j, cls] = np.bincount(rows[:, j], minlength=16)

        total = counts.sum(axis=1, keepdims=True)
        logp = np.log((counts + ALPHA) / (total + K * ALPHA))
        tabs[name] = np.rint(logp * SCORE_SCALE).astype(np.int16)

    return tabs


def predict_integer_lut(tabs, test_idx, custom_q=None):
    source = Q if custom_q is None else custom_q
    scores = np.zeros((len(test_idx), K), dtype=np.int64)

    for name in SELECTED:
        A = source[name][test_idx]
        table = tabs[name]
        for j in range(A.shape[1]):
            scores += np.take(table[j], A[:, j], axis=1).T

    return scores.argmax(axis=1)


def accuracy(a, b):
    return float(np.mean(a == b))


split_results = []

for seed in SPLIT_SEEDS:
    tr, te = train_test_split(
        np.arange(len(y)),
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    tabs = train_integer_lut(tr, y[tr])
    mprc = accuracy(predict_integer_lut(tabs, te), y[te])

    gd_raw = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2500, solver="lbfgs", random_state=seed),
    )
    gd_raw.fit(GRAW[tr], y[tr])
    gd_raw_acc = accuracy(gd_raw.predict(GRAW[te]), y[te])

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=float)
    qtr = enc.fit_transform(GQ[tr])
    qte = enc.transform(GQ[te])

    gd_qpair = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=seed,
        )
    )
    gd_qpair.fit(qtr, y[tr])
    gd_qpair_acc = accuracy(gd_qpair.predict(qte), y[te])

    split_results.append(
        {
            "seed": seed,
            "mprc_integer_lut": mprc,
            "gd_raw_channels": gd_raw_acc,
            "gd_qpair_onehot": gd_qpair_acc,
        }
    )


def mean(key):
    return float(np.mean([r[key] for r in split_results]))


def sample_std(key):
    return float(np.std([r[key] for r in split_results], ddof=1))


# Canonical seed-42 controls.
tr, te = train_test_split(
    np.arange(len(y)),
    test_size=0.30,
    random_state=42,
    stratify=y,
)

normal_tabs = train_integer_lut(tr, y[tr])

# Random-label control.
rng = np.random.default_rng(20260923)
random_train_labels = y[tr].copy()
rng.shuffle(random_train_labels)

random_tabs = train_integer_lut(tr, random_train_labels)
random_label_accuracy = accuracy(
    predict_integer_lut(random_tabs, te),
    y[te],
)

# Independent per-image shuffle preserves each image's histogram while destroying shape.
shuffled = np.empty_like(U8)
for i, img in enumerate(U8):
    rr = np.random.default_rng(880000 + i)
    flat = img.reshape(-1).copy()
    rr.shuffle(flat)
    shuffled[i] = flat.reshape(8, 8)

shuffled_channels = [channels(im) for im in shuffled]
shuffled_q = {
    name: np.stack([qpair(c[name]) for c in shuffled_channels])
    for name in SELECTED
}

shuffled_tabs = {}
for name in SELECTED:
    A = shuffled_q[name][tr]
    counts = np.zeros((A.shape[1], K, 16), dtype=np.int32)
    for cls in range(K):
        rows = A[y[tr] == cls]
        for j in range(A.shape[1]):
            counts[j, cls] = np.bincount(rows[:, j], minlength=16)
    total = counts.sum(axis=1, keepdims=True)
    shuffled_tabs[name] = np.rint(
        np.log((counts + ALPHA) / (total + K * ALPHA)) * SCORE_SCALE
    ).astype(np.int16)

pixel_shuffle_accuracy = accuracy(
    predict_integer_lut(shuffled_tabs, te, shuffled_q),
    y[te],
)


def shifted_images(dx, dy):
    shifted = np.zeros_like(U8)

    for i, img in enumerate(U8):
        rr0 = max(0, dy)
        rr1 = min(8, 8 + dy)
        cc0 = max(0, dx)
        cc1 = min(8, 8 + dx)

        sr0 = max(0, -dy)
        sr1 = sr0 + (rr1 - rr0)
        sc0 = max(0, -dx)
        sc1 = sc0 + (cc1 - cc0)

        shifted[i, rr0:rr1, cc0:cc1] = img[sr0:sr1, sc0:sc1]

    return shifted


translation = {}
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    shifted = shifted_images(dx, dy)
    shifted_channels = [channels(im) for im in shifted]
    shifted_q = {
        name: np.stack([qpair(c[name]) for c in shifted_channels])
        for name in SELECTED
    }

    translation[f"{dx},{dy}"] = accuracy(
        predict_integer_lut(normal_tabs, te, shifted_q),
        y[te],
    )

rotated = np.rot90(U8, k=1, axes=(1, 2)).copy()
rotated_channels = [channels(im) for im in rotated]
rotated_q = {
    name: np.stack([qpair(c[name]) for c in rotated_channels])
    for name in SELECTED
}
rotation90_accuracy = accuracy(
    predict_integer_lut(normal_tabs, te, rotated_q),
    y[te],
)

result = {
    "architecture": {
        "channels": SELECTED,
        "qpair_edges": len(EDGES),
        "alpha": ALPHA,
        "integer_score_scale": SCORE_SCALE,
    },
    "repeated_splits": split_results,
    "summary": {
        key: {
            "mean": mean(key),
            "sample_std": sample_std(key),
        }
        for key in (
            "mprc_integer_lut",
            "gd_raw_channels",
            "gd_qpair_onehot",
        )
    },
    "controls": {
        "random_label_accuracy": random_label_accuracy,
        "pixel_shuffle_accuracy": pixel_shuffle_accuracy,
        "translation_test_accuracy": translation,
        "rotation90_test_accuracy": rotation90_accuracy,
    },
}

output = ROOT / "results" / "gap_closure_digits.json"
output.write_text(json.dumps(result, indent=2), encoding="utf-8")

print(json.dumps(result, indent=2))