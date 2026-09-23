# Canonical MPRC Vision Manifold

The canonical Arshad-ViT/MPRC vision hypervector is:

```text
(112 + 16 channel rows) x (112 + 1 polarity column)
= 128 x 113
= 14,464 uint8 states
```

The 112x112 image payload is preserved exactly. The 1,920 additional states are structural storage, not replacement pixels.

## Physical layout

| Region | Shape | States | Meaning |
|---|---:|---:|---|
| Pixel payload | 112x112 | 12,544 | Original image bytes |
| Channel storage | 16x112 | 1,792 | One structural row per channel |
| Polarity storage | 112x1 | 112 | One polarity structural dimension |
| Channel x polarity corner | 16x1 | 16 | Channel/polarity intersection |
| **Structural total** |  | **1,920** | |
| **Whole manifold** | 128x113 | **14,464** | |

The 16 channel identities are:

```text
R, G, B, gray, luma, chroma,
gx, gy, grad, lap,
h1, h2, m4, l8, contrast, orient
```

This file freezes the **storage geometry only**.

It does **not** invent how a 112x112 channel plane is reduced or encoded into a 112-state structural row, nor how the polarity column is populated. Those value semantics belong to the metadata codec and must be derived/tested separately.

The key invariant is:

```text
pixels remain pixels
metadata remains metadata
```

A learning experiment that hides absolute position inside a classifier table index is therefore only a control, not the canonical manifold implementation.
