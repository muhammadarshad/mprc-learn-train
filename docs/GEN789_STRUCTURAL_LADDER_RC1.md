# GEN 7-8-9 Structural Ladder — Closure Note RC1

**Status:** exact identities + one conditional uniqueness theorem  
**Date:** 26 September 2026  
**Scope:** foundation mathematics only; no benchmark claim.

This note records the structural relation noticed while auditing

\[
113=1+2(7)(8)
\]

and

\[
252=4(7)(9)=4(3+4)9.
\]

The key point is that these are not two unrelated numerical decompositions.
They come from one consecutive ladder around the half-square root of the
\(D=256\) state domain.

---

## 1. Base definitions

Let

\[
D=256,
\qquad
s=\frac{\sqrt D}{2}.
\]

For \(D=256\),

\[
s=8.
\]

The canonical MPRC generator is

\[
GEN=s-1=7.
\]

Hence the consecutive ladder is

\[
\boxed{GEN,\ GEN+1,\ GEN+2 = 7,8,9.}
\]

Also

\[
N=\sqrt D-1=15.
\]

Since

\[
\sqrt D=2(GEN+1),
\]

we obtain the exact identity

\[
\boxed{N=2GEN+1.}
\]

For \(GEN=7\),

\[
N=15.
\]

---

## 2. The 113 identity is the same object in two coordinate systems

The canonical manifold width is frozen as

\[
W=2GEN^2+N.
\]

Substitute

\[
N=2GEN+1.
\]

Then

\[
W
=
2GEN^2+2GEN+1
=
\boxed{1+2GEN(GEN+1)}.
\]

For \(GEN=7\),

\[
W
=
1+2(7)(8)
=
113.
\]

But the integer Manhattan ball of radius \(r\) in a five-site cross geometry
has cardinality

\[
|B_r|
=
1+2r(r+1).
\]

Therefore at \(r=GEN\),

\[
\boxed{
W
=
2GEN^2+N
=
1+2GEN(GEN+1)
=
|B_{GEN}|.
}
\]

For the frozen values,

\[
\boxed{
113
=
2(7^2)+15
=
1+2(7)(8)
=
|B_7|.
}
\]

This is an exact identity, not an empirical observation.

### Equivalent \(D\)-form

Because

\[
GEN=\frac{\sqrt D}{2}-1,
\]

we also have

\[
W
=
1+2
\left(\frac{\sqrt D}{2}-1\right)
\left(\frac{\sqrt D}{2}\right)
\]

and hence

\[
\boxed{
W
=
\frac D2-\sqrt D+1.
}
\]

For \(D=256\),

\[
W=128-16+1=113.
\]

Thus the frozen DATA+INFORMATION width formula and the seven-round five-arm
dependency count are algebraically identical.

### Boundary qualification

The count \(113\) is the full dependency set for a site whose radius-7
neighborhood is not truncated by a slab or lane boundary.

It is a dependency/support count, not a guarantee that 113 output values are
numerically nonzero after modular cancellation.

---

## 3. The 252 identity comes from the same 7-8-9 ladder

The existing MPRC count is

\[
M_3=4(7)(9).
\]

Using the ladder,

\[
M_3
=
4GEN(GEN+2).
\]

Now

\[
GEN(GEN+2)
=
(GEN+1)^2-1.
\]

Therefore

\[
M_3
=
4\left((GEN+1)^2-1\right).
\]

Since

\[
GEN+1=\frac{\sqrt D}{2},
\]

we obtain

\[
M_3
=
4\left(\frac D4-1\right)
=
\boxed{D-4}.
\]

Thus

\[
\boxed{
4GEN(GEN+2)=D-4.
}
\]

For \(D=256\),

\[
\boxed{
4(7)(9)=256-4=252.
}
\]

Equivalently, preserving the owner's original decomposition,

\[
\boxed{
252=4(3+4)9.
}
\]

Under the existing four-vacuum MPRC convention, this is exactly the active
state count

\[
ACTIVE=D-4=252.
\]

The algebraic identity \(4GEN(GEN+2)=D-4\) is proved here.
The semantic interpretation of the four excluded states as vacuums belongs to
the existing MPRC construction and is not re-proved by this note.

---

## 4. The fixed phase encoder lands on the same 252

The owner-supplied 16-channel encoder defines Phase by

\[
Phase
=
\left(
R\bmod7
+
G\bmod7
+
B\bmod7
\right)14.
\]

Each residue is in

\[
\{0,1,\ldots,6\}.
\]

Hence the maximum is

\[
Phase_{\max}
=
(6+6+6)14
=
18\cdot14
=
252.
\]

Now factor it:

\[
18=2\cdot9,
\qquad
14=2\cdot7.
\]

Therefore

\[
Phase_{\max}
=
(2\cdot9)(2\cdot7)
=
\boxed{4(7)(9)}
=
252.
\]

So the fixed encoder has the exact numerical closure

\[
\boxed{
Phase_{\max}
=
4GEN(GEN+2)
=
D-4
=
252.
}
\]

This is an exact identity for the current encoder.

It does **not** by itself prove that the semantic role of the Phase channel is
"ACTIVE-state enumeration"; that stronger interpretation remains open.

---

## 5. Conditional uniqueness theorem for the generalized phase family

The current encoder is fixed at modulus 7 and multiplier 14.

To test whether 7 is accidental, define the following **candidate generalized
family**:

\[
\Phi_g(R,G,B)
=
\left(
R\bmod g
+
G\bmod g
+
B\bmod g
\right)(2g).
\]

This family is a research construction introduced only for the uniqueness
test. It is not part of the frozen encoder definition.

Its maximum is

\[
\Phi_{\max}(g)
=
3(g-1)(2g)
=
\boxed{6g(g-1)}.
\]

From the structural ladder associated with generator \(g\), define

\[
A(g)=4g(g+2).
\]

Ask when the generalized phase maximum closes exactly onto the structural
active count:

\[
6g(g-1)=4g(g+2).
\]

For positive \(g\), divide by \(2g\):

\[
3(g-1)=2(g+2).
\]

Hence

\[
3g-3=2g+4
\]

and therefore

\[
\boxed{g=7}.
\]

So, **within this explicitly stated candidate family**,

\[
\boxed{
\Phi_{\max}(g)=A(g)
\iff
g=7
}
\]

for positive integer \(g\).

This is a conditional uniqueness theorem.

Since

\[
D(g)=4(g+1)^2,
\]

the unique solution \(g=7\) gives

\[
D=4(8^2)=256.
\]

Thus the generalized closure condition simultaneously selects

\[
\boxed{GEN=7,\qquad D=256.}
\]

Again: the uniqueness result is rigorous **conditional on the candidate family**
\(\Phi_g\). The family itself is not claimed as a frozen MPRC law.

---

## 6. Compact structural form

Let

\[
s=\frac{\sqrt D}{2},
\qquad
GEN=s-1.
\]

Then the three consecutive factors are

\[
\boxed{s-1,\ s,\ s+1.}
\]

The two observed counts are

\[
\boxed{
W=1+2(s-1)s
}
\]

and

\[
\boxed{
ACTIVE=4(s-1)(s+1).
}
\]

For \(D=256\), \(s=8\):

\[
\boxed{
7,\ 8,\ 9
}
\]

with

\[
\boxed{
W=1+2(7)(8)=113
}
\]

and

\[
\boxed{
ACTIVE=4(7)(9)=252.
}
\]

So the same ladder produces:

- \(7\times8\) in the full seven-round five-arm support/manifold-width count;
- \(7\times9\) in the \(M_3\)/active-state count.

This is the foundation relation to preserve.

---

## 7. Status table

| Statement | Status |
|---|---|
| \(N=2GEN+1\) | EXACT IDENTITY |
| \(W=2GEN^2+N=1+2GEN(GEN+1)\) | EXACT IDENTITY |
| \(W=|B_{GEN}|\) for the five-site Manhattan dependency geometry | THEOREM |
| \(4GEN(GEN+2)=D-4\) | EXACT IDENTITY |
| \(4(7)(9)=252\) | EXACT IDENTITY |
| fixed encoder \(Phase_{\max}=(6+6+6)14=252\) | EXACT IDENTITY |
| fixed encoder \(Phase_{\max}=4GEN(GEN+2)\) at \(GEN=7\) | EXACT IDENTITY |
| generalized \(\Phi_g=(\sum RGB\bmod g)(2g)\) | CANDIDATE CONSTRUCTION |
| \(\Phi_{\max}(g)=4g(g+2)\iff g=7\) | CONDITIONAL THEOREM |
| Phase semantics equal ACTIVE-state semantics | OPEN / NOT PROVED |

---

## 8. Research consequence

The correct next question is no longer merely

> why do 113 and 252 keep appearing?

The stronger question is

\[
\boxed{
\text{Why does the MPRC substrate organize around }
(s-1,s,s+1)
=
(7,8,9)?
}
\]

because both

\[
113
\]

and

\[
252
\]

are exact projections of that same consecutive structural ladder.

This note closes the arithmetic relationship.
It does not yet close the observation-to-structural packing map \(\Pi\),
ReactionLUT semantics, IDENTIFY, or U/MOVE.
