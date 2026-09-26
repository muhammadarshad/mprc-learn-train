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

## Exact composition seam still unresolved

The frozen sources establish independently:

- QH4 addresses byte/ring values exactly;
- directional ADI-9 is exact;
- generator-7 is a lossless ring/address traversal;
- rectangular transpose is bijective;
- BIND -> REACT -> MEASURE is defined exactly.

What is **not yet specified tightly enough in the current source** is the state-transition
rule that makes **QH4 + generator-7 navigation alter/feed the directional ADI manifold**
before BIND/REACT, rather than merely being returned as metadata/diagnostics.

That rule must be sourced or stated explicitly. It must not be invented to improve CIFAR
accuracy.

## Next legal step

Implement no training.

First implement an instrumented `ArshadBlock` only after the missing transport-composition
rule is frozen. Its forward trace must show nonzero call counts and byte counts for:

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
