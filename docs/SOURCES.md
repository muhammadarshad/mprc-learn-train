# Source specifications

This research repo keeps implementation and benchmark code separate from the source specifications.

## Arshad's ViT

Source supplied by the author: **Arshad's ViT — Self-Contained Coding Specification v1.0**.

Frozen implementation points used here:

- native ring: `Z_256`
- generator: `7`
- local vision identity: `3x3 = 1 reference + 8 ADI relations`
- attention path: `IDENTIFY -> BIND -> REACT -> MEASURE`
- 64-address stride-7 transport
- 5-site `S5` REACT topology

The benchmark in this repo uses only the local ADI-9, stride-7, and S5 observables needed for the empirical ablations.

## QH4 VOF interface reconstruction

Source supplied by the author: **Volume-of-Fluid Interface Reconstruction on the QH4 Ring**.

The vision ablation transfers only one explicitly tested primitive from that paper:

- integer second differences, used there for curvature.

The full PLIC/VOF solver, orientation transport, and conservation claims are not relabeled as vision operators here.

## Public MPRC book

https://muhammadarshad.github.io/pages-mprc/

Chapter 22 is the ADI reference used in the structural experiments.
