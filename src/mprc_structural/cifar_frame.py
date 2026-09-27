"""Lossless CIFAR32 -> MPRC observation/manifold framing.

No resizing.  The 32x32 source is centered in the fixed 113x128 observation
frame.  Added frame state is 255, deliberately not one of the QH4 vacua
{0,64,128,192}.  Each RGB channel is then transposed to the execution shape
128x113.

The inverse extracts exactly the original 32x32x3 source bytes.
"""

from __future__ import annotations
import numpy as np

OBS_H=113
OBS_W=128
SRC_H=32
SRC_W=32
CHANNELS=3
FILL=255
VACUUM=frozenset((0,64,128,192))
AY=(OBS_H-SRC_H)//2
AX=(OBS_W-SRC_W)//2
assert FILL not in VACUUM
assert AY==40 and AX==48


def frame_rgb32(rgb: np.ndarray) -> np.ndarray:
    x=np.asarray(rgb)
    if x.shape!=(SRC_H,SRC_W,CHANNELS) or x.dtype!=np.uint8:
        raise ValueError("expected uint8 [32,32,3]")
    out=np.full((CHANNELS,OBS_H,OBS_W),FILL,dtype=np.uint8)
    out[:,AY:AY+SRC_H,AX:AX+SRC_W]=np.transpose(x,(2,0,1))
    return out


def to_manifolds(observation: np.ndarray) -> np.ndarray:
    x=np.asarray(observation)
    if x.shape!=(CHANNELS,OBS_H,OBS_W) or x.dtype!=np.uint8:
        raise ValueError("expected uint8 [3,113,128]")
    return np.transpose(x,(0,2,1)).copy()


def from_manifolds(manifolds: np.ndarray) -> np.ndarray:
    x=np.asarray(manifolds)
    if x.shape!=(CHANNELS,OBS_W,OBS_H) or x.dtype!=np.uint8:
        raise ValueError("expected uint8 [3,128,113]")
    return np.transpose(x,(0,2,1)).copy()


def recover_rgb32(observation: np.ndarray) -> np.ndarray:
    x=np.asarray(observation)
    if x.shape!=(CHANNELS,OBS_H,OBS_W) or x.dtype!=np.uint8:
        raise ValueError("expected uint8 [3,113,128]")
    src=x[:,AY:AY+SRC_H,AX:AX+SRC_W]
    return np.transpose(src,(1,2,0)).copy()
