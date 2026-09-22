# Sunday Freeze Constraints — 20 September 2026 (RC2)

This file records the mathematical boundaries that empirical learning work in this repository must respect.

## Paper 1 — Quantinion Simplex Cubic Decomposition

- The theorem concerns an `(n+1)`-term symmetric cubic decomposition whose parity columns form a binary simplex frame.
- For `(n,m)=(8,8)`, the frozen specialization has 9 cubic components in dimension 8 over `Z256`.
- This is an identifiability theorem for that simplex class, not a generic prescription for image features.

## Paper 2 — Exact Free Cubic Dictionary Capacity

For symmetric rank-one cubic dictionaries over `Z/2^m Z`:

`Cap_3(n,m) = sum_{k=1}^{min(3,n)} C(n,k)`.

For `n=8`:

`Cap_3(8,m) = 8 + 28 + 56 = 92`.

Important boundary: 92 is the exact free dictionary capacity. It is not a claim that arbitrary 92-term unknown-atom decompositions are identifiable, and it is not permission to call any arbitrary 92 handcrafted image features "the cubic dictionary".

## Paper 3 — Eight-State Operator / Transpose Geometry

- Eight states are indexed by one reference state plus seven non-reference states.
- `Q={1,2,4} subset Z7` canonically orients the 21 unordered pairs.
- The trace-zero operator module has the exact free decomposition
  `63 = 14 + 21 + 28 = 56 + 7`.
- The active QH4 address is exactly
  `m = 64 theta + 21 a + 7 p + sigma`, followed by `z = 7m mod 256`.
- The local Transpose theorem is proved only for its stated local `3x3` coordinates.
- Canonical tensor-lift kernels over `Z256`:
  - order 2: `{0,128 I}`
  - order 3: `{0}`
  - order 4: `{0,64 I,128 I,192 I}`

Therefore order two has the central half-turn ambiguity while the canonical third-order action is faithful. This algebraic result must not be silently generalized to a different tensor action.

## Paper 4 — Excess Two

For full-row-rank parity frame `S in F2^{n x (n+2)}`, with relation code `K=ker S`:

`global (n+2)-term identifiability <=> d(K) >= 5`.

Equivalently, no nonempty relation on 1, 2, 3, or 4 parity atoms may vanish.

## Paper 5 — Excess Three

For `S in F2^{n x (n+3)}`, `K=ker S`:

`ker kappa_S = 0 <=> d(K) >= 5`.

The proof uses generalized Hamming weights and a finite twenty-orbit GF(2) certificate after analytic localization.

## Empirical policy

The following are experimental unless separately proved:

- any observation-to-operator coefficient map;
- any image classifier built from the frozen operator basis;
- any use of cubic/shadow coordinates as vision features;
- LUT scoring laws;
- channel selection;
- robustness/generalization claims.

A benchmark gain never upgrades an empirical construction into one of the five frozen theorems.
