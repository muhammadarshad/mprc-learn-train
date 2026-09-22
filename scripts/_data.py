from pathlib import Path
import numpy as np

def load_vendored(root: Path):
    d=np.load(root / "data/digits_seed42.npz")
    X=d["X"].astype(np.float64).reshape(-1,8,8)
    y=d["y"].astype(np.int64)
    train=d["train_idx"].astype(np.int64)
    test=d["test_idx"].astype(np.int64)
    return X,y,train,test
