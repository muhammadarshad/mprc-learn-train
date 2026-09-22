from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_laplace

ORIGIN = 128
SIGMA = 0.8

OFF8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def log_batch(images: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    return np.stack([gaussian_laplace(im, sigma=sigma, mode="nearest") for im in images])

def fit_z256_scale(log_train: np.ndarray) -> float:
    s = float(np.max(np.abs(log_train)))
    return s if s != 0 else 1.0

def to_z256(log_arr: np.ndarray, scale: float) -> np.ndarray:
    q = np.rint((log_arr / scale) * 127.0).astype(np.int16)
    q = np.clip(q, -127, 127)
    z = np.where(q > 0, ORIGIN-q, np.where(q < 0, ORIGIN+(-q), ORIGIN))
    return z.astype(np.uint8)

def hv_edges_8x8():
    edges = []
    for r in range(8):
        for c in range(7):
            edges.append((r*8+c, r*8+c+1))
    for r in range(7):
        for c in range(8):
            edges.append((r*8+c, (r+1)*8+c))
    return edges

def orbit7_64():
    out = []
    q = 0
    for _ in range(64):
        out.append(q)
        q = (q + 7) & 63
    assert len(set(out)) == 64
    return out

def stride7_edges():
    o = orbit7_64()
    return [(o[t], o[(t+1) % 64]) for t in range(64)]

def feature_pixels(z: np.ndarray):
    return [int(v) for v in z.reshape(-1)]

def feature_adi9(z: np.ndarray):
    vals = []
    for r in range(1,7):
        for c in range(1,7):
            center = int(z[r,c])
            neigh = [int(z[r+dr,c+dc]) for dr,dc in OFF8]
            lam = (center + sum(neigh)) & 255
            vals.append(lam)
            vals.extend([(center-n) & 255 for n in neigh])
    return vals

def feature_walk7(z: np.ndarray):
    f = z.reshape(-1)
    o = orbit7_64()
    return [(int(f[o[t]]) - int(f[o[(t+1)%64]])) & 255 for t in range(64)]

def feature_vof_second_diff(z: np.ndarray):
    vals = []
    for r in range(1,7):
        for c in range(1,7):
            cc = int(z[r,c])
            dx2 = (int(z[r,c-1]) + int(z[r,c+1]) - 2*cc) & 255
            dy2 = (int(z[r-1,c]) + int(z[r+1,c]) - 2*cc) & 255
            vals.extend([dx2, dy2])
    return vals

def feature_s5(z: np.ndarray):
    vals = []
    for r in range(1,7):
        for c in range(1,7):
            s = (
                int(z[r,c]) + int(z[r-1,c]) + int(z[r+1,c]) +
                int(z[r,c-1]) + int(z[r,c+1])
            ) & 255
            vals.append(s)
    return vals
