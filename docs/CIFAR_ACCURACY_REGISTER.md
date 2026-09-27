# CIFAR / Arshad-ViT Accuracy Register

Target: **>94% CIFAR-10 classification accuracy**.

This register tracks empirical classifier results only. It does **not** redefine
the frozen Arshad-ViT algebra.

## Frozen algebra boundary

Canonical computation remains:

- QCM manifold: **128 x 113 = 14,464 computational Z256 state/address slots**
- execution: **two 64 x 113 slabs**
- generator transport: **q_t = q_0 + 7*t (mod 64)**
- local structure: **3 x 3 = 1 reference + 8 ADI relations**
- native byte geometry: **16 x 7 <-> 7 x 16**
- attention: **IDENTIFY -> BIND -> REACT -> MEASURE**
- canonical REACT topology: **7 rounds -> 113-site support**

Source pixels, computational states/materialized arrays, and physical storage
accounting are separate quantities. A storage/packing observation must never
change the computational geometry or operators.

## Empirical classifier lineage before the storage-accounting divergence

| Model | Validation | Shadow / holdout | Test / exploratory | Status |
|---|---:|---:|---:|---|
| MPRC Vision CIFAR v0 | 45.72% | n/a | **46.68%** | strongest early classifier baseline |
| v1 directional 1+8 | 45.52% | 44.88% | 45.42% | empirical regression |
| Geometry A/B directional | 45.52% | 44.88% | 45.42% | empirical regression |
| Geometry A/B square | 45.42% | 45.92% | 45.99% | empirical regression |
| ADI9 exact branch | 10.34% | n/a | 10.39% | routing/representation probe only |
| Arm8 W2/B1 | 18.12% | 18.72% | 19.03% | probe only |
| Active-anchor Arm8 | 19.30% | 18.34% | 19.96% | probe only |
| Position/displacement Arm8 | 22.54% | 20.94% | 22.51% | probe only |
| Exact post-survival SELECT | 3.18% all / 19.88% covered | 3.86% all | 3.52% all | SELECT diagnostic |

## Quarantined post-divergence results

The following experiments were based on the invalid reinterpretation of the
frozen 128x113 computational manifold as a 16x113 / Z16 execution geometry.
They remain in Git history for audit but are **not evidence about canonical
Arshad-ViT**:

- corrected byte-state attention / Z16 generator transport
- repeated byte-state CIFAR stream
- capacity-aligned A0 113x16 classifier
- A1 900-center capacity-pass classifier

The reported A0 result (47.06% shadow) is therefore quarantined and must not be
used as a promoted ViT baseline.

The spectral CXR A/B remains a **control experiment only**. It does not define
or modify the ViT algebra.

## Current valid empirical baseline

Until a new classifier is run on the restored frozen algebra, the strongest
valid CIFAR classifier result remains:

**MPRC Vision CIFAR v0: 46.68% exploratory official-test accuracy.**

## Recovery rule

Do not change frozen constants or operators to repair benchmark performance.
Diagnose only unfrozen interfaces such as routing policy, MetadataCodec,
ReactionLUT/training rule, classifier head, or composition of the already
frozen operators.
