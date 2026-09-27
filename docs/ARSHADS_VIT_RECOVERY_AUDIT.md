# Arshad-ViT Recovery Audit — 2026-09-27

## Purpose

This document records the recovery after a storage/accounting correction was
allowed to mutate the frozen ViT execution algebra.

## Last safe research boundary

**Commit:** `fa2da2128bb07413e2e95a26973530cda5cca26a`  
**Message:** `Run coherent-displacement SELECT survival gate`

At this point the repository still followed the frozen Arshad-ViT execution
geometry and operator sequence.

## First divergence

**Commit:** `674422cb71d7b90b88c7f0ef7b573a3423a675cd`  
**Message:** `Correct ViT manifold to 14,464-bit storage`

The intent was to correct byte-vs-pixel/storage accounting. The accounting
distinction is useful, but it must not redefine the computational manifold.

## Algebra-breaking divergence

**Commit:** `8eac65b4becf5855db75ce7957508f2726699983`  
**Message:** `Correct REACT to 1,808-byte computational state`

This changed canonical execution from:

- 128x113 computational Z256 states
- two 64x113 slabs
- generator-7 transport on Z64

to a non-canonical:

- 16x113 execution state
- generator-7 transport on Z16

That change is rejected.

## Restored canonical equations

The frozen coding spec remains authoritative:

[
R = Z_{256}
]

[
H 	imes W = 128 	imes 113
]

[
q_t = q_0 + 7t pmod{64}
]

[
3	imes3 = 1 + 8
]

[
16	imes7 leftrightarrow 7	imes16
]

[
IDENTIFY ightarrow BIND ightarrow REACT ightarrow MEASURE
]

[
support(7)=113
]

The typed execution candidate remains:

[
IDENTIFY_{ADI9/QH4}
ightarrow TYPED TRANSPOSE
ightarrow GENERATOR7 TRANSPORT
ightarrow BIND
ightarrow REACT
ightarrow MEASURE
]

SELECT and MOVE remain downstream routing/decision stages; they are not reasons
to alter the frozen operator algebra.

## Accounting distinction

The recovery keeps the useful distinction:

- **source pixels**: coordinates/bytes read from the source image
- **computational states**: frozen QCM/GEVHV state/address slots and any
  materialized derived arrays
- **physical storage**: implementation-specific packed representation

No equality between these categories may be used to change the frozen equations.

The optional `bit_manifold.py` utility is now explicitly quarantined as a
packing/accounting helper only. It is not a canonical execution module.

## Removed from current master

The following invalid branches were removed from active source/workflows while
remaining available in Git history:

- `src/mprc_structural/byte_attention.py`
- `gates/gate_byte_attention.py`
- byte-attention survival workflow
- corrected byte-state CIFAR streaming experiment/workflow
- capacity-aligned A0 experiment/workflow
- full900 A1 capacity-pass experiment/workflow

## Still valid / retained

- `docs/ARSHADS_VIT_SPEC_EXTRACT.md`
- `src/mprc_structural/manifold.py`
- `src/mprc_structural/attention.py`
- `src/mprc_structural/typed_attention.py`
- QH4, directional ADI9, 16x7/7x16 transpose
- original pre-training exactness gates
- batch/scalar attention parity gates
- typed IDENTIFY / transpose survival
- coherent-displacement SELECT survival
- earlier loss-localization experiments, especially the finding that collapsing
  pair information before/through BIND+REACT can destroy useful signal

## Empirical classifier status

Post-divergence A0/A1 numbers are quarantined because their architecture depended
on the invalid Z16 reinterpretation.

The strongest valid historical CIFAR classifier baseline remains the pre-divergence
MPRC Vision CIFAR v0 result of 46.68% exploratory official-test accuracy.

The >94% objective remains, but future progress must be achieved by completing
and tuning unfrozen parts of the frozen ViT architecture, not by changing its
constants or state geometry.
