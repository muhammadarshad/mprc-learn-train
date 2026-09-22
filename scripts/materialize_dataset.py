from pathlib import Path
import hashlib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "digits_seed42.npz"

digits = load_digits()
X = digits.data.astype(np.uint8)
y = digits.target.astype(np.uint8)

idx = np.arange(len(y))
train_idx, test_idx = train_test_split(
    idx, test_size=0.30, random_state=42, stratify=y
)

np.savez_compressed(
    OUT,
    X=X,
    y=y,
    train_idx=train_idx.astype(np.uint16),
    test_idx=test_idx.astype(np.uint16),
)

h = hashlib.sha256()
h.update(X.tobytes())
h.update(y.tobytes())
h.update(train_idx.astype(np.uint16).tobytes())
h.update(test_idx.astype(np.uint16).tobytes())

print("wrote", OUT)
print("canonical_payload_sha256", h.hexdigest())
print("expected", "86cddf309a5e475f8b918426113b0312faa5b4830e612a956d3be8859f488270")
assert h.hexdigest() == "86cddf309a5e475f8b918426113b0312faa5b4830e612a956d3be8859f488270"
