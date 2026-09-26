# v24 — Observation to Structural Interface

**Status:** foundation candidate + exact interface theorems  
**Purpose:** close the coordinate/type seam between the 16-channel wave observer and the canonical 128x113 structural manifold without inventing semantic metadata.

---

## 1. Two state types

The fixed wave observer produces

\[
\mathcal O
=
R^{16\times113\times128},
\qquad
R=\mathbb Z_{256}.
\]

One structural manifold has type

\[
\mathcal S
=
R^{128\times113}.
\]

Therefore the 16-channel observation bank may be viewed as sixteen equal-cardinality frames:

\[
\mathcal O
=
\left(R^{113\times128}\right)^{16}.
\]

Each channel contains exactly

\[
113\cdot128
=
14,464
\]

ring states.

---

## 2. Candidate frame-interface definition

Define the channel-wise frame transpose

\[
\boxed{
\Pi_T:
\mathcal O
\to
\mathcal S^{16}
}
\]

by

\[
\boxed{
(\Pi_T\Psi)_{c,h,w}
=
\Psi_{c,w,h}.
}
\]

Thus

\[
16\times113\times128
\longrightarrow
16\times128\times113.
\]

No channel is averaged, collapsed or mixed.

This is a **DEFINITION / CANDIDATE INTERFACE**.

The definition does not claim that ordinary frame transpose is the unique MPRC packing law.

---

## 3. Exact theorem: \(\Pi_T\) is a bijection

Applying the same transpose twice returns the input:

\[
\boxed{
\Pi_T^{-1}=\Pi_T.
}
\]

Therefore

\[
\boxed{
\Pi_T(\Pi_T(\Psi))=\Psi.
}
\]

No byte/state is lost.

Per channel:

\[
113\cdot128
=
128\cdot113
=
14,464.
\]

Across all sixteen channels:

\[
16\cdot14,464
=
231,424
\]

states are preserved exactly.

---

## 4. Exact theorem: ring BIND commutes with \(\Pi_T\)

For coordinate-wise ring BIND

\[
BIND(A,B)=A\oplus B,
\]

\[
\boxed{
\Pi_T(A\oplus B)
=
\Pi_T(A)\oplus\Pi_T(B).
}
\]

So \(\Pi_T\) is an additive module isomorphism for the coordinate-wise \(Z_{256}\) state law.

---

## 5. Exact theorem: MEASURE is invariant under \(\Pi_T\)

For wide ring energy

\[
MEASURE(A,B)
=
\sum_u d(A_u,B_u),
\]

the transpose only permutes coordinates.

Therefore

\[
\boxed{
MEASURE(\Pi_TA,\Pi_TB)
=
MEASURE(A,B).
}
\]

This holds independently per channel and for the sum over all channels.

Thus \(\Pi_T\) cannot create or destroy evidence under coordinate-wise MEASURE.

---

## 6. Compatibility with native 16x7 <-> 7x16 rectangles

The frozen byte geometry uses

\[
16\times7
\leftrightarrow
7\times16.
\]

A global coordinate transpose has exactly this local restriction.

For any valid source block

\[
A[r:r+7,\;c:c+16],
\]

the corresponding target block is

\[
(\Pi_TA)[c:c+16,\;r:r+7]
=
A[r:r+7,\;c:c+16]^T.
\]

So every local

\[
7\times16
\]

rectangle maps exactly to

\[
16\times7,
\]

and vice versa.

This is an exact compatibility theorem between the candidate frame interface and the frozen native rectangle transpose.

It does not prove uniqueness of \(\Pi_T\).

---

## 7. The 98+15 partition after transpose

Each target channel has shape

\[
128\times113.
\]

Therefore it admits the canonical structural coordinate partition

\[
113=98+15.
\]

So coordinates can be typed as

\[
\boxed{
S_c=
(DATA_c,\ INFORMATION_c)
}
\]

with

\[
DATA_c\in R^{128\times98},
\]

\[
INFORMATION_c\in R^{128\times15}.
\]

This statement is only a **coordinate partition**.

It does NOT prove that the values landing in the last 15 columns already have the required semantic meaning of MPRC INFORMATION.

That semantic problem remains open.

---

## 8. Correct DATA / INFORMATION orientation

The total structural state budget is fixed:

\[
\boxed{M=14,464.}
\]

The INFORMATION quantity is

\[
\boxed{I=1,920.}
\]

It is **not appended** to the full manifold.

Instead the preserved DATA quantity is obtained by subtraction:

\[
\boxed{
D=M-I
}
\]

so

\[
\boxed{
D=14,464-1,920=12,544.
}
\]

This matches the canonical construction

\[
\boxed{
14,464=12,544+1,920.
}
\]

The important orientation is therefore

\[
\boxed{
M \rightarrow (M-I,\ I)
}
\]

not

\[
M+I.
\]

So the previous "independent extra metadata" reading was the wrong model for
the owner's construction.

---

## 9. Bidirectional structural action of the 1,920 quantity

The INFORMATION quantity is structurally available in both directions:

\[
\boxed{
M\pm I.
}
\]

For the frozen values,

\[
M-I
=
14,464-1,920
=
\boxed{12,544},
\]

while the opposite branch is

\[
M+I
=
14,464+1,920
=
16,384.
\]

The canonical preserved DATA branch is the **negative** direction:

\[
\boxed{
DATA=M-I=12,544.
}
\]

Thus \(1,920\) is not being interpreted as additional independent storage.
It is a structural displacement/reserve relative to the fixed manifold count.

The semantic meaning of the positive branch remains open unless separately
defined by the MPRC construction.

---

## 10. Important observation about the 16-channel encoder

The actual encoder is deterministic from three RGB source channels and preserves RGB directly as channels 0-2.

Therefore the 16 output channels are not sixteen independent arbitrary random variables.

The map

\[
RGB\to\Psi_{16}
\]

is injective because RGB is included in the output.

So the true degrees of freedom of the encoder output are at most those of its RGB source:

\[
3\times14,464
\]

byte states, even though the materialized representation contains

\[
16\times14,464
\]

states.

This redundancy may eventually support a multi-channel structural codec.

However, no such codec is frozen yet.

---

## 11. What v24 closes

If the finite gate passes, v24 closes:

\[
\boxed{\text{G6a: coordinate/type interface}}
\]

as the candidate

\[
\Pi_T:
R^{16\times113\times128}
\leftrightarrow
(R^{128\times113})^{16}.
\]

It proves:

- exact byte preservation;
- exact inverse;
- BIND compatibility;
- MEASURE invariance;
- compatibility with native \(16\times7\leftrightarrow7\times16\) rectangles.

---

## 12. What remains open

v24 does NOT close:

### G6b — INFORMATION semantics

What do the 15 INFORMATION lanes mean after observation?

### G6c — multi-channel reduction/binding

Do the sixteen manifolds remain parallel, or is there an exact invertible relation/binding among them?

### 15+1 channel interpretation

The numerical relation

\[
16=15+1
\]

remains structurally suggestive but is not yet identified with the 15 INFORMATION lanes.

### G7 onward

ReactionLUT, IDENTIFY, U/MOVE, SELECT and resolved memory remain separate open interfaces.

---

## 13. Scaling boundary

A full experiment must not silently do

\[
16\text{ channels}
\to
1\text{ manifold}
\]

by averaging, pooling, arbitrary channel selection, flatten/reshape or learned projection.

Any such map must be explicit, typed, and either invertible or deliberately information-reducing with a stated claim boundary.
