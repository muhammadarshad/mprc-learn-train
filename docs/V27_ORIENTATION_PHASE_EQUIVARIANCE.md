# v27 — Orientation-Phase Equivariance

**Status:** exact execution theorem for the v24 candidate interface.  
**No dataset, labels, or benchmark tuning are used.**

Let (T) be the v24 channel-wise coordinate transpose:

[
T: 113\times128 \leftrightarrow 128\times113.
]

The frozen operators satisfy:

[
T(BIND(A,Q))=BIND(TA,TQ),
]

because BIND is coordinate-wise addition over (Z_{256}).

For the five-site isotropic staple,

[
S_5(X)_{ij}=X_{ij}+X_{i-1,j}+X_{i+1,j}+X_{i,j-1}+X_{i,j+1},
]

transpose exchanges row and column neighbours but preserves the same five-term sum. Therefore for any single shared 256-entry LUT (L),

[
T(REACT_L(X))=REACT_L(TX).
]

Circular MEASURE is a sum over coordinates, so

[
MEASURE(TA,TQ)=MEASURE(A,Q).
]

Hence a row/column schedule that alternates the v24 orientation between consecutive REACT rounds does not introduce a learnable phase choice. With seven REACT rounds there are six between-round transposes, so

[
(T R)^6 R
]

reduces to the same final orientation and the same byte state as (R^7), because (TR=RT) and (T^2=I).

The code gate verifies this bit-exactly on the full 128x113 / 113x128 shapes using identity and nonlinear LUTs.

## Consequence

For this candidate interface, H/V phase alternation is an **execution/layout symmetry**, not an accuracy knob. It must not be tuned on CIFAR.

This theorem does **not** close:

1. the missing rule by which QH4 + corrected directional ADI IDENTIFY changes or selects the state/query entering BIND;
2. task-specific ReactionLUT learning.

Those remain pre-training blockers.
