# IMPLEMENTATION GAP REPORT — Directional Arshad's ViT

Status: **TRAINING BLOCKED**

Authority order used here:

1. frozen Arshad's ViT specification / principal research prompt
2. verified mathematics
3. current code
4. old experiments

## Corrections now frozen

The local context is not a conventional 3x3 square-neighbour mask.

The ordered nine-byte local state is

```
C, U1, U2, D1, D2, F1, F2, B1, B2
```

i.e.

```
C + 2*UP + 2*DOWN + 2*FORWARD + 2*BACKWARD = 1 + 8 = 9
```

Its ring-native ADI representation is

```
Lambda = sum(all 9 bytes) mod 256
delta_k = C - arm_k mod 256
```

Since `9^-1 mod 256 = 57`, recovery is exact:

```
C = 57 * (Lambda + sum(delta_1..delta_8)) mod 256
arm_k = C - delta_k mod 256
```

No 512-state scalar packing is permitted.

## Exactness status

| Gate | Status |
|---|---|
| directional ADI-9 exact inverse | PASS |
| first-depth S5 recoverable from ADI | PASS |
| second-depth directional continuation recoverable | PASS |
| odd-unit affine BIND covariance | PASS |
| QH4 forward/inverse 252/252 | PASS |
| generator 7 / inverse 183 | PASS |
| 16x7 <-> 7x16 byte transpose | PASS |
| directional ADI covariance under transpose | PASS |
| 7-round 5-site dependency support = 113 | PASS |
| cdist 65,536 byte pairs | PASS |

Machine-readable results are emitted by:

- `gates/gate_directional_adi9.py`
- `gates/gate_directional_backbone.py`

## Current code audit

### BLOCKER 1 — old local geometry

The existing reference `local_descriptor()` still calls the old local ADI implementation
for the conventional immediate 3x3 neighbourhood. It does not implement the corrected
two-depth 5-arm context.

### BLOCKER 2 — generator-7 is diagnostic, not inference

The reference constructor creates `self.walk = chunk_walk2d(...)`, but
`forward_manifold()` does not pass that walk into `attention_forward()`.
It returns `chunk_walk` as diagnostics after the forward call.

Therefore the current code does **not** prove generator-7 participates in inference.

### BLOCKER 3 — QH4 absent from actual reference forward

The current reference forward does not call the QH4 forward/inverse address functions.
QH4 exists as verified mathematics/utilities, but utility existence does not satisfy the
frozen requirement that the real ArshadBlock execute it.

### BLOCKER 4 — rectangular transpose absent from actual reference forward

The frozen 16x7 <-> 7x16 transport is not called by the current
`forward_manifold()` state path.

### BLOCKER 5 — multires QH4 coverage report is not a measurement

An existing multires audit sets `qh4_coverage_pct = 100.0` directly. That value is not
an executed coverage measurement and must not be used as evidence.

### BLOCKER 6 — task-specific reaction learning is open

The reference topology explicitly leaves task-specific LUT learning/selection outside v0.
A training procedure must define this state before B5 training.

## Quarantined experiments

The existing CIFAR v0/v1/geometry-A-B classifier experiments are marked:

**QUARANTINED — PRE-SURVIVAL / NONCANONICAL**

Reasons:

1. they were trained before the full forward graph survived;
2. v0 used the wrong square-neighbour interpretation;
3. v1 packed nine binary observations into a 512-state scalar rather than a Z256 ADI state;
4. the QH4 seed states byte atomicity; bit-plane classifiers are not accepted as the
   canonical MPRC byte path.

Their numbers remain historical diagnostics only.

## Exact composition seam — narrowed by Paper 3

The frozen sources now establish the following exact bridge in addition to the earlier
primitive gates:

```
gamma - 1 = 9*theta + 3*a + p
m = 64*theta + 21*a + 7*p + sigma
z = 7*m mod 256
```

with `theta in Z4`, `a,p in Z3`, and `sigma=1..7`. This is the exact
`4 x 9 x 7 = 252` QH4 active-state decomposition from Paper 3.

Paper 3 also proves the local Arshad Transpose

```
T_k(a,p) = (p+k, a-k) mod 3
```

with `T_k^2=I`, invariant `a+p`, and 84 fixed + 84 two-cycles globally.

These facts close the **address-level** coupling between the 9 local slots, 7 transport
steps, four quarters, QH4 active addresses, and the local transpose orbit geometry.

Two semantic/runtime seams remain and must NOT be guessed:

1. The corrected directional order
   `(C,U1,U2,D1,D2,F1,F2,B1,B2)` has not yet been given a frozen semantic
   bijection to Paper-3's `(a,p)` slot coordinate.
2. Paper-3 `T_k` and native image H<->V transpose are now proved to be
   **different operators**. The corrected directional spatial transpose has cycle type
   `1^1 2^4`; every Paper-3 `T_k` has cycle type `1^3 2^3`. They cannot be
   identified by any bijection.
3. The native `16x7 <-> 7x16` transpose is proved to commute with coordinate-wise
   BIND and REACT (for arbitrary pointwise LUT) and to leave MEASURE invariant when
   state/query transpose together. Therefore the GEVHV core itself has no unresolved
   H/V arithmetic.
4. What remains open is the **IDENTIFY/routing phase semantics**: which orientation
   phase is selected by QH4/directional identity and when that routing choice changes.
   It is not a BIND/REACT/MEASURE correctness problem.

Thus the principal remaining C1/C2 seam is now the directional IDENTIFY -> QH4/routing
interface, not the native transpose or GEVHV core arithmetic.

## Next legal step

Implement no training.

First implement an instrumented `ArshadBlock` only after the remaining directional-slot
mapping and runtime phase schedule are frozen. Its forward trace must show nonzero call counts and byte counts for:

```
QH4
directional ADI-9
generator-7 transport
16x7 <-> 7x16 transport
BIND
REACT
MEASURE
```

Only after that trace and exactness gates pass may CIFAR training begin.


---

## Current closure update — v24/v27/v28

This section supersedes the older blocker wording above without deleting the audit history.

### C2 — H/V phase schedule: CLOSED for the v24 candidate interface

v27 proves for the v24 channel-wise full-frame transpose that:

- BIND commutes with transpose;
- the isotropic five-site REACT commutes with transpose for any shared 256-entry LUT;
- MEASURE is transpose invariant;
- seven REACT rounds with six between-round H/V transposes are bit-identical to seven rounds without those transposes.

Therefore H/V starting phase and between-round alternation are not benchmark-tunable parameters under this interface.

Status: **EXACT EXECUTION SYMMETRY / CLOSED FOR v24**.

### C3 — INFORMATION semantics: narrowed, not an execution blocker for v24 observation frames

v24 maps the fixed 16-channel observation bank

    16 x 113 x 128

bijectively to

    16 x 128 x 113.

All 231,424 states are preserved. The 98+15 target partition exists coordinate-wise, but semantic interpretation of the last 15 lanes remains open.

For the v24 observation path, no synthetic metadata values are invented: the full observed frame is carried through. Therefore missing semantic names for the 15 lanes do not block byte-exact execution.

Status:

- execution/storage interface: **CLOSED CANDIDATE, EXACT BIJECTION**
- semantic interpretation: **OPEN CLAIM BOUNDARY**

### C4 — ReactionLUT: v28 candidate under survival

v28 fixes, before benchmark training, a label-free candidate objective:

    L[u] = argmin_y sum_c H[u,c] * cdist(y,c)

where H[u,c] counts observed center bytes c for five-site staple input u.

Each LUT entry is a finite independent 256-candidate circular-L1 medoid problem. The rule is integer-only, byte-native and observation-populated.

Status: **PENDING CI SURVIVAL**. It is not accepted until the v28 exact gate passes.

### Remaining hard composition blocker

C1 remains:

    QH4 content address + corrected directional ADI evidence
        -> decision/query influence before attention readout

The wider MPRC source clarifies that QH4 locate() depends on byte value, not sequential/spatial position. Therefore QH4 must not be silently reinterpreted as a pixel-coordinate permutation.

Directional ADI supplies local spatial relation; QH4 supplies content/ring address. Their typed combination is valid, but the rule that makes that IDENTIFY evidence influence candidate/query selection is still not frozen.

**Training remains blocked until C1 is closed and v28 passes.**
