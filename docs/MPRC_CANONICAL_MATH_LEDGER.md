# MPRC Canonical Math Ledger

**Status:** canonical memory aid / research ledger  
**Date:** 26 September 2026  
**Purpose:** preserve important MPRC mathematics in one place so the research does not depend on remembering old chats.

This file is a compact index of currently frozen or carefully classified mathematics.

Every statement must be tagged as one of:

- **THEOREM** — proved in the stated domain.
- **EXACT IDENTITY** — algebraically exact from frozen definitions.
- **DEFINITION** — chosen construction/rule.
- **CONDITIONAL THEOREM** — rigorous after explicitly stated assumptions/construction.
- **CONJECTURE / OPEN** — not yet proved.
- **EMPIRICAL** — observed in experiments only.

Do not upgrade an EMPIRICAL result into a theorem because a benchmark passes.

---

# 1. Ring substrate

## 1.1 Base domain — DEFINITION

\[
\boxed{R=\mathbb Z_{256}}
\]

with

\[
\boxed{\tau=256,\quad O=128,\quad Q=64,\quad GEN=7.}
\]

Ring addition/subtraction:

\[
a\oplus b=(a+b)\bmod256,
\]

\[
a\ominus b=(a-b)\bmod256.
\]

Circular distance:

\[
\boxed{
d(a,b)=
\min\left((a-b)\bmod256,\;(b-a)\bmod256\right).
}
\]

The implemented distance is a metric on the 256-cycle.

---

# 2. Canonical manifold

## 2.1 Derived constants — EXACT IDENTITIES

Let

\[
D=256.
\]

Then

\[
\sqrt D=16,
\]

\[
\boxed{N=\sqrt D-1=15},
\]

\[
\boxed{GEN=\frac{\sqrt D}{2}-1=7},
\]

\[
\boxed{H=\frac D2=128}.
\]

DATA width:

\[
\boxed{W_D=2GEN^2=98}.
\]

INFORMATION width:

\[
\boxed{W_I=N=15}.
\]

Total width:

\[
\boxed{W=W_D+W_I=113}.
\]

Therefore

\[
\boxed{
M=H\,W=128\cdot113=14,464.
}
\]

Byte partition:

\[
\boxed{DATA=128\cdot98=12,544}
\]

and

\[
\boxed{INFORMATION=128\cdot15=1,920}.
\]

Thus

\[
\boxed{
14,464=12,544+1,920.
}
\]


### 2.2 INFORMATION orientation — EXACT CONSTRUCTION

The \(1,920\) INFORMATION quantity is **inside** the fixed \(14,464\)-state budget. It is not appended as an independent extra object.

The preserved DATA quantity is

\[
\boxed{
DATA=M-I=14,464-1,920=12,544.
}
\]

The structural quantity \(I=1,920\) may be considered arithmetically in both directions,

\[
M\pm I,
\]

but the canonical data-preserving branch is

\[
\boxed{
M-I=12,544.
}
\]

The \(M+I=16,384\) branch is not assigned semantic meaning here unless separately defined.

Also

\[
\boxed{GEN^2+N=49+15=64}.
\]

---



# 2A. INFORMATION polarity dual-budget envelope

The structural envelope also admits the exact count

\[
\boxed{
E=(GEN^2+N)D=64\times256=16,384.
}
\]

The half-domain INFORMATION budget is

\[
\boxed{
I_{1/2}=N\frac D2=15\times128=1,920.
}
\]

The full-domain INFORMATION budget is

\[
\boxed{
I_1=ND=15\times256=3,840=2I_{1/2}.
}
\]

Therefore

\[
\boxed{
E-I_{1/2}=16,384-1,920=14,464=M
}
\]

and

\[
\boxed{
E-I_1=16,384-3,840=12,544=DATA.
}
\]

Equivalently,

\[
\boxed{
16,384
\to
14,464
\to
12,544
}
\]

in two equal steps of

\[
\boxed{-1,920}.
\]

Also,

\[
\boxed{
DATA=GEN^2D=49\times256=12,544.
}
\]

**Implementation boundary:** the \(15\times128\) branch is represented in current code. The \(15\times256\) branch has not yet been exercised because the current implementation does not instantiate a \(64\times256\) execution state.

# 3. GEN 7-8-9 structural ladder

Define

\[
s=\frac{\sqrt D}{2}.
\]

For \(D=256\),

\[
s=8.
\]

Since

\[
GEN=s-1,
\]

the consecutive structural ladder is

\[
\boxed{GEN,\;GEN+1,\;GEN+2=7,8,9.}
\]

Also

\[
N=\sqrt D-1=2s-1.
\]

Because \(s=GEN+1\),

\[
\boxed{N=2GEN+1.}
\]

For \(GEN=7\),

\[
N=15.
\]

---

# 4. Why 113 appears

The canonical width is

\[
W=2GEN^2+N.
\]

Using

\[
N=2GEN+1,
\]

we get

\[
W
=
2GEN^2+2GEN+1
=
\boxed{1+2GEN(GEN+1)}.
\]

For \(GEN=7\),

\[
\boxed{
113
=
2(7^2)+15
=
1+2(7)(8).
}
\]

## 4.1 Five-site REACT geometry — THEOREM

For the Manhattan ball

\[
B_r=\{(x,y):|x|+|y|\le r\},
\]

the exact number of sites is

\[
\boxed{|B_r|=1+2r(r+1)}.
\]

Therefore at

\[
r=GEN,
\]

\[
\boxed{
|B_{GEN}|
=
1+2GEN(GEN+1)
=
W.
}
\]

For \(GEN=7\),

\[
\boxed{|B_7|=113.}
\]

So

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

Equivalent \(D\)-form:

\[
\boxed{
W=\frac D2-\sqrt D+1.
}
\]

For \(D=256\),

\[
W=128-16+1=113.
\]

**Boundary note:** the \(113\) count is the full dependency/support set for an interior radius-7 five-site neighborhood. It does not guarantee 113 numerically nonzero outputs because modular cancellation may occur.

---

# 5. Why 252 appears

Define the structural \(M_3\) / active-state count

\[
\boxed{
M_3(g)=4g(g+2).
}
\]

For \(g=GEN=7\),

\[
\boxed{
M_3=4(7)(9)=252.
}
\]

Preserving the owner's form:

\[
\boxed{
252=4(3+4)9.
}
\]

Now

\[
g(g+2)=(g+1)^2-1.
\]

Therefore

\[
4g(g+2)
=
4\left((g+1)^2-1\right).
\]

For the canonical family

\[
D=4(g+1)^2,
\]

we get

\[
\boxed{
4g(g+2)=D-4.
}
\]

Hence at \(D=256\),

\[
\boxed{
ACTIVE=D-4=252.
}
\]

Under the existing MPRC convention, the four excluded states are

\[
\{0,64,128,192\}.
\]

The algebraic identity

\[
4GEN(GEN+2)=D-4
\]

is exact.

---

# 6. Fixed 16-channel wave encoder

The fixed owner-supplied encoder includes the Phase channel

\[
\boxed{
Phase=
\left[
(R\bmod7)+(G\bmod7)+(B\bmod7)
\right]14.
}
\]

Each residue is in

\[
\{0,\ldots,6\}.
\]

Therefore

\[
Phase_{\max}
=
(6+6+6)14.
\]

Now

\[
6+6+6=18=2\cdot9
\]

and

\[
14=2\cdot7.
\]

Hence

\[
\boxed{
Phase_{\max}
=
(6+6+6)14
=
(2\cdot9)(2\cdot7)
=
4(7)(9)
=
252.
}
\]

Therefore the fixed encoder has the exact numerical closure

\[
\boxed{
Phase_{\max}
=
4GEN(GEN+2)
=
D-4
=
252
}
\]

at

\[
GEN=7,\quad D=256.
\]

**Status:** EXACT IDENTITY.

**Not proved:** this arithmetic equality alone does not prove that the semantic meaning of the Phase channel equals the semantic meaning of the ACTIVE-state set.

---

# 7. Candidate generalized phase family

To test whether \(g=7\) is accidental, define the candidate family

\[
\boxed{
\Phi_g(R,G,B)
=
\left[
(R\bmod g)+(G\bmod g)+(B\bmod g)
\right](2g).
}
\]

This is a **CANDIDATE CONSTRUCTION**, not a frozen MPRC law.

Its maximum is

\[
\Phi_{\max}(g)
=
3(g-1)(2g)
=
\boxed{6g(g-1)}.
\]

Define

\[
\boxed{
P(g)=6g(g-1)=6g^2-6g.
}
\]

Structural capacity:

\[
\boxed{
M(g)=4g(g+2)=4g^2+8g.
}
\]

Set them equal:

\[
6g(g-1)=4g(g+2).
\]

Rearrange:

\[
2g^2-14g=0
\]

so

\[
\boxed{
2g(g-7)=0.
}
\]

Thus the roots are

\[
g=0,\quad g=7.
\]

For positive generators,

\[
\boxed{
P(g)=M(g)
\iff
g=7.
}
\]

This is a **CONDITIONAL THEOREM** within the stated candidate generalized family.

At \(g=7\),

\[
D=4(g+1)^2
=
4(8^2)
=
256.
\]

So the conditional closure simultaneously selects

\[
\boxed{
GEN=7,\qquad D=256.
}
\]

---

# 8. Residual / balance polynomial

Define the difference

\[
\boxed{
F(g)=P(g)-M(g).
}
\]

Then

\[
F(g)
=
6g(g-1)-4g(g+2)
\]

\[
=
2g^2-14g
\]

\[
=
\boxed{2g(g-7)}.
\]

This is the central balance polynomial.

At the positive root,

\[
\boxed{
F(7)=0.
}
\]

Therefore

\[
\boxed{
P(7)=M(7)=252.
}
\]

For positive \(g\):

- \(0<g<7\): \(F(g)<0\), so structural capacity exceeds phase maximum.
- \(g=7\): exact balance.
- \(g>7\): \(F(g)>0\), so phase maximum exceeds structural capacity.

---

# 9. Continuous first and second derivatives

Treat \(g\) temporarily as a real variable.

Phase family:

\[
P(g)=6g^2-6g.
\]

Therefore

\[
\boxed{
P'(g)=12g-6
}
\]

and

\[
\boxed{
P''(g)=12.
}
\]

Structural family:

\[
M(g)=4g^2+8g.
\]

Therefore

\[
\boxed{
M'(g)=8g+8
}
\]

and

\[
\boxed{
M''(g)=8.
}
\]

Difference:

\[
F(g)=P(g)-M(g).
\]

Therefore

\[
\boxed{
F'(g)=4g-14
}
\]

and

\[
\boxed{
F''(g)=4.
}
\]

At \(g=7\),

\[
P'(7)=78,
\]

\[
M'(7)=64,
\]

so

\[
\boxed{
F'(7)=78-64=14.
}
\]

And since

\[
2g=14
\]

at \(g=7\),

\[
\boxed{
F'(7)=2g=14.
}
\]

## 9.1 Why \(F'(7)=2g\) is forced

Start from

\[
F(g)=2g(g-7).
\]

Differentiate using the product rule:

\[
F'(g)
=
2(g-7)+2g.
\]

At the root \(g=7\), the first term vanishes:

\[
F'(7)=2g.
\]

Therefore the appearance of \(2g=14\) in the velocity gap at the crossing is algebraically forced by the same factorization.

It is structurally meaningful, but it should not be counted as an independent proof of \(g=7\).

---

# 10. Continuous acceleration gap

Because

\[
P''(g)=12
\]

and

\[
M''(g)=8,
\]

the acceleration difference is

\[
\boxed{
F''(g)=12-8=4.
}
\]

This means the continuous velocity gap increases by an **additive constant 4** per unit \(g\).

Use the phrase:

\[
\boxed{\text{constant additive acceleration gap of }4}
\]

not "factor of 4".

---

# 11. Discrete generation differences

MPRC generation \(g\) is naturally integer-valued, so discrete differences are important.

Define the forward difference

\[
\Delta F(g)=F(g+1)-F(g).
\]

For

\[
F(g)=2g^2-14g,
\]

we obtain

\[
\boxed{
\Delta F(g)=4g-12.
}
\]

At \(g=7\),

\[
\boxed{
\Delta F(7)=16.
}
\]

Define the backward difference

\[
\nabla F(g)=F(g)-F(g-1).
\]

Then

\[
\boxed{
\nabla F(g)=4g-16.
}
\]

At \(g=7\),

\[
\boxed{
\nabla F(7)=12.
}
\]

For a quadratic, the centered difference

\[
\frac{F(g+1)-F(g-1)}{2}
\]

equals the continuous derivative exactly:

\[
\boxed{
\frac{F(g+1)-F(g-1)}{2}=4g-14.
}
\]

At \(g=7\),

\[
\boxed{
12,\ 14,\ 16
}
\]

appear as

\[
\boxed{
2(g-1),\ 2g,\ 2(g+1)
}
\]

because

\[
2(6)=12,\quad
2(7)=14,\quad
2(8)=16.
\]

Thus the local discrete growth-gap ladder around the balance point is

\[
\boxed{
2(g-1),\;2g,\;2(g+1).
}
\]

At \(g=7\),

\[
\boxed{
12,\;14,\;16.
}
\]

---

# 12. Second finite difference

For a quadratic residual,

\[
\Delta^2F(g)
=
F(g+2)-2F(g+1)+F(g).
\]

Since the quadratic coefficient of \(F\) is \(2\),

\[
\boxed{
\Delta^2F(g)=4
}
\]

for every integer \(g\).

Thus the discrete and continuous second-order gaps agree:

\[
\boxed{
F''(g)=\Delta^2F(g)=4.
}
\]

This is exact.

---

# 13. Compact hierarchy at the balance point

At the canonical balance point

\[
g=7,
\]

we have:

### State/capacity level

\[
\boxed{
P(7)=M(7)=252.
}
\]

### Continuous velocity-gap level

\[
\boxed{
F'(7)=14=2g.
}
\]

### Local discrete velocity-gap level

\[
\boxed{
\nabla F(7)=12,
\quad
F'_{\text{center}}(7)=14,
\quad
\Delta F(7)=16.
}
\]

Equivalently,

\[
\boxed{
12,14,16
=
2(6,7,8)
=
2(g-1,g,g+1).
}
\]

### Acceleration-gap level

\[
\boxed{
F''(7)=\Delta^2F(7)=4.
}
\]

So the current exact hierarchy is

\[
\boxed{
\begin{aligned}
F(g)&=2g(g-7),\\
F'(g)&=4g-14,\\
F''(g)&=4,\\
\nabla F(7)&=12,\\
F'_{\text{center}}(7)&=14,\\
\Delta F(7)&=16.
\end{aligned}
}
\]

---

# 14. Structural ladders currently known

## 14.1 Primary GEN ladder

\[
\boxed{
g-1,\ g,\ g+1,\ g+2
}
\]

around \(g=7\) gives

\[
\boxed{
6,\ 7,\ 8,\ 9.
}
\]

Important projections:

\[
\boxed{
113=1+2(7)(8)
}
\]

and

\[
\boxed{
252=4(7)(9).
}
\]

The discrete velocity-gap ladder is

\[
\boxed{
12,14,16=2(6,7,8).
}
\]

So the values \(6,7,8,9\) now appear in three exact places:

1. Phase residues use \(0\ldots6\).
2. GEN is \(7\).
3. The manifold support identity uses \(7\cdot8\).
4. The active-state identity uses \(7\cdot9\).

The arithmetic statements are exact.

The deeper semantic interpretation of the whole \(6,7,8,9\) ladder is still **OPEN**.

---

# 15. Status ledger

| Statement | Status |
|---|---|
| \(R=\mathbb Z_{256}\), origin 128, GEN=7 | DEFINITION / FROZEN |
| \(N=2GEN+1\) | EXACT IDENTITY |
| \(W=2GEN^2+N\) | FROZEN CONSTRUCTION |
| \(W=1+2GEN(GEN+1)\) | EXACT IDENTITY |
| \(|B_r|=1+2r(r+1)\) | THEOREM |
| \(W=|B_{GEN}|\) | THEOREM |
| \(4GEN(GEN+2)=D-4\) | EXACT IDENTITY |
| \(4(7)(9)=252\) | EXACT IDENTITY |
| fixed \(Phase_{\max}=(6+6+6)14=252\) | EXACT IDENTITY |
| candidate \(\Phi_g=(\sum RGB\bmod g)(2g)\) | CANDIDATE CONSTRUCTION |
| \(P(g)=6g(g-1)\) for candidate family | EXACT WITHIN CANDIDATE |
| \(M(g)=4g(g+2)\) | STRUCTURAL FAMILY |
| \(F(g)=2g(g-7)\) | EXACT IDENTITY |
| positive balance \(P=M\iff g=7\) | CONDITIONAL THEOREM |
| \(F'(g)=4g-14\) | EXACT CONTINUOUS DERIVATIVE |
| \(F''(g)=4\) | EXACT CONTINUOUS DERIVATIVE |
| \(F'(7)=14=2g\) | EXACT IDENTITY |
| \(\nabla F(7)=12,\ F'_c(7)=14,\ \Delta F(7)=16\) | EXACT DISCRETE IDENTITIES |
| \(\Delta^2F=4\) | EXACT DISCRETE IDENTITY |
| Phase semantics = ACTIVE-state semantics | OPEN |
| 6-7-8-9 ladder has one physical interpretation | OPEN |
| 16 channels = 15 INFORMATION + 1 completion state | CONJECTURE / OPEN |
| observation-to-structural packing map \(\Pi\) | OPEN |

---

# 16. Foundation interfaces still open

The arithmetic is not yet a complete MPRC machine.

Still open:

\[
\boxed{
\Pi:
\mathbb Z_{256}^{16\times113\times128}
\rightarrow
\text{canonical structural state}
}
\]

plus:

- admissible ReactionLUT law;
- IDENTIFY state type;
- U/MOVE temporal state;
- SELECT contract;
- resolved-memory contract;
- one end-to-end finite theorem using all of the above.

---

# 17. Files that protect this ledger

Relevant frozen/verification files:

- \`docs/FOUNDATION_CLOSURE_RC1.md\`
- \`docs/GEN789_STRUCTURAL_LADDER_RC1.md\`
- \`docs/MANIFOLD_16C_1P.md\`
- \`docs/SUNDAY_FREEZE_RC2.md\`
- \`scripts/foundation_closure_gate_rc1.py\`
- \`scripts/gen789_structural_ladder_gate.py\`
- \`scripts/wave_encoder_torch.py\`

This ledger is intended to be the first place to look when a future discussion asks:

> "What math have we actually closed?"
