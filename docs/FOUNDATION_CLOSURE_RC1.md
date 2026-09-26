# MPRC Foundation Closure RC1

**Status:** foundation audit / scaling gate  
**Date:** 26 September 2026  
**Purpose:** stop empirical version-chasing until the mathematical interfaces needed for a full experiment are explicit.

The repository already contains strong frozen pieces, but they do not yet form one closed end-to-end mathematical machine.

The required discipline is:

\[
\boxed{\text{prove/define the substrate} \to \text{close interfaces} \to \text{finite gates} \to \text{scale experiment}}
\]

No benchmark result can close a missing interface.

---

## 0. Four status words

Every statement in the scaled system must be tagged as one of:

- **THEOREM** — proved in its stated domain.
- **DEFINITION** — chosen mathematical object/rule; internally exact but not a theorem about nature or learning.
- **CONJECTURE** — mathematically precise claim not yet proved.
- **EMPIRICAL** — observation from code/data.

Do not promote EMPIRICAL -> THEOREM because a benchmark wins.

---

## 1. What is already closed

### F1. Ring substrate — DEFINITION + elementary theorem layer

Base state space:

\[
R=\mathbb Z/256\mathbb Z.
\]

Frozen constants:

\[
\tau=256,\qquad O=128,\qquad Q=64,\qquad GEN=7.
\]

Addition/subtraction:

\[
a\oplus b=(a+b)\bmod 256,
\qquad
a\ominus b=(a-b)\bmod256.
\]

Circular distance:

\[
d(a,b)=\min((a-b)\bmod256,(b-a)\bmod256).
\]

Closed properties:

\[
0\le d(a,b)\le128,
\quad
d(a,b)=d(b,a),
\quad
d(a,b)=0\iff a=b.
\]

The finite gate exhaustively checks the triangle inequality on all \(256^3\)
triples, therefore the implemented \`cdist\` is a metric on the 256-cycle.

---

### F2. GEN7 transport — THEOREM

Because

\[
\gcd(7,64)=\gcd(7,256)=1,
\]

addition by 7 generates complete cycles on both \(\mathbb Z_{64}\) and
\(\mathbb Z_{256}\).

The exact inverses are

\[
7^{-1}\equiv55\pmod{64},
\qquad
7^{-1}\equiv183\pmod{256}.
\]

Thus generator transport is a coordinate permutation, not information loss.

For any common coordinate permutation \(T\),

\[
T(A\oplus B)=T(A)\oplus T(B)
\]

and wide ring MEASURE is permutation invariant:

\[
MEASURE(TA,TB)=MEASURE(A,B).
\]

Therefore GEN7 transport by itself cannot create evidence. Its mathematical
role is to define the neighborhood geometry on which REACT operates.

---

### F3. Canonical DATA + INFORMATION manifold — THEOREM/CONSTRUCTION

From \(D=256\),

\[
\sqrt D=16,\quad N=15,\quad GEN=7,\quad H=128.
\]

Widths:

\[
W_D=2GEN^2=98,
\qquad
W_I=N=15,
\qquad
W=113.
\]

Counts:

\[
DATA=128\cdot98=12,544,
\]

\[
INFORMATION=128\cdot15=1,920,
\]

\[
M=128\cdot113=14,464.
\]

And

\[
GEN^2+N=49+15=64.
\]

Execution:

\[
128\times113=2(64\times113).
\]

This is modality-agnostic. It is not, by itself, an image geometry.

---

### F4. BIND — DEFINITION + homomorphism

For equal-shaped states,

\[
BIND(A,B)=A\oplus B
\]

coordinate-wise in \(R\).

It is exact, closed in \(R\), and commutes with common coordinate
permutations.

---

### F5. MEASURE — DEFINITION + metric sum

For states \(A,B\) over the same finite coordinate set \(\Omega\),

\[
MEASURE(A,B)=\sum_{u\in\Omega} d(A_u,B_u).
\]

Because \(d\) is a metric, this finite sum is also a metric on equal-shaped
state arrays.

Use a wide integer accumulator. The score itself is not a byte.

---

## 2. REACT: what is closed and what is not

The current five-site topology in generator-logical coordinates is

\[
L(X)_{i,j}
=
X_{i,j}
+X_{i-1,j}
+X_{i+1,j}
+X_{i,j-1}
+X_{i,j+1}
\pmod{256}
\]

for interior sites, with boundary sites preserved.

### F6. Identity-LUT REACT — THEOREM

With the identity LUT, the implemented REACT operator is additive:

\[
L(A\oplus B)=L(A)\oplus L(B).
\]

Thus it is a homomorphism of the finite additive state module.

### F7. Seven-round dependency cone — THEOREM

A five-site cross expands graph dependency by Manhattan distance one per round.

After \(r\) rounds, an interior site can depend on the Manhattan ball

\[
B_r=\{(x,y):|x|+|y|\le r\}.
\]

Its exact cardinality is

\[
|B_r|=1+2r(r+1).
\]

At \(r=7\),

\[
\boxed{|B_7|=1+2(7)(8)=113.}
\]

This is a **dependency/support-capacity theorem**, not a guarantee that all 113
numeric outputs are nonzero; modular cancellation can occur.

### OPEN R1. ReactionLUT law

The repository currently permits an arbitrary 256-entry LUT.

Once a nonlinear LUT is inserted, additivity need not survive.

Before scaling, the LUT family must be frozen mathematically:

\[
LUT:R\to R
\]

with explicit required properties, e.g. whether it must preserve origin,
polarity, be bijective, be monotone in ring distance, or satisfy another MPRC
law.

Until that family is specified, "learned REACT" is EMPIRICAL.

---

## 3. The 16-channel wave observer is mathematically exact — but its interface is not closed

The owner-supplied observer is a fixed map

\[
E:
R^{3\times113\times128}
\longrightarrow
R^{16\times113\times128}.
\]

The 16 channels are:

\[
(R,G,B,\text{Luma},\text{Gray},\text{Chroma},\text{Phase},\text{Winding},
d_1,d_2,\text{LoG}_0,\ldots,\text{LoG}_{128}).
\]

No learnable parameter is required by this encoder.

### F8. Encoder codomain closure — THEOREM

For byte inputs:

- \(R,G,B\in[0,255]\).
- Luma remains in \([0,255]\) because \(77+150+29=256\).
- Gray remains in \([0,255]\).
- Chroma is \(\max-\min\in[0,255]\).
- Phase has maximum \((6+6+6)14=252\).
- Winding is in \(\{0,85,170\}\).
- \(d_1,d_2\) are explicitly reduced modulo 256.
- One heat step is

\[
H(x)=\left\lfloor
\frac{N+S+W+E+4C}{8}
\right\rfloor
\]

on the interior. The numerator lies in \([0,2040]\), so \(H(x)\in[0,255]\).
- The plaquette is explicitly reduced modulo 256.

Hence every output channel is closed in \(R\).

This proves range/integer closure. Parity with separate NumPy/Rust
implementations is a code-equivalence claim and should retain its own parity
test.

---

## 4. Critical interface gap: observation frame != structural manifold yet

This is the main blocker to a full experiment.

The encoder produces

\[
16\times113\times128.
\]

The structural attention code consumes one state shaped

\[
128\times113
\]

with the semantic partition

\[
98\ DATA + 15\ INFORMATION.
\]

The equal cardinality

\[
113\cdot128=128\cdot113=14,464
\]

does **not** by itself define semantic equivalence.

A coordinate swap

\[
\phi(r,c)=(c,r)
\]

is a trivial bijection between the two index rectangles, but it does not tell
us:

1. whether each of the 16 channels is one complete \(14,464\)-state manifold;
2. how the 98 DATA lanes and 15 INFORMATION lanes are populated;
3. whether the 16 channels remain separate through attention;
4. how channels interact under BIND/REACT;
5. whether some 15+1 relationship is intended between the 16 observer channels
   and the structural \(N=15\) construction.

### OPEN I1. Packing map

Before scaling we need an explicit typed map

\[
\boxed{\Pi:\mathcal O\to\mathcal S}
\]

from observation state

\[
\mathcal O=R^{16\times113\times128}
\]

to the exact structural state type \(\mathcal S\) used by attention.

No experiment should silently substitute a reshape, transpose, average,
feature pooling, or channel selection for \(\Pi\).

### CONJECTURE C1 — 15+1 / 16 relation

The numerical relation

\[
16=15+1=\sqrt{256}
\]

is structurally suggestive, but **no theorem in the current freeze identifies
the 16 observer channels with the 15 INFORMATION lanes plus one completion
state**.

Do not use that identification until it is explicitly defined and proved or
adopted as a construction.

---

## 5. Missing state-transition foundation

The current scaled story wants

\[
OBSERVE
\to IDENTIFY
\to BIND
\to REACT
\to MEASURE
\to MOVE
\to SELECT
\to MEMORY.
\]

The middle algebra is partly closed; the endpoints are not.

### OPEN I2. IDENTIFY

Required:

\[
IDENTIFY:\mathcal O\to\mathcal F
\]

where \(\mathcal F\) is an explicitly defined factor/evidence state space.

"Shape", "color", "motion", etc. cannot be silently hard-coded into arbitrary
feature vectors and then called MPRC IDENTIFY.

Channel selection is currently EMPIRICAL under the Sunday freeze.

### OPEN I3. MOVE / U-Observer

Need exact state variables and transition law, not a dataset-specific search.

At minimum freeze types for

\[
X_t,\quad \Delta X_t,\quad V_t,\quad U_t
\]

and define which are stored, which are derived, and how \(U\) changes from
observations through time.

The synthetic movement experiments validate mechanics only; they do not close
this definition.

### OPEN I4. SELECT

Need a deterministic mathematical contract.

Candidate form:

\[
SELECT(C,q)=\arg\min_{c\in C} MEASURE(c,q)
\]

is exact only after the candidate set \(C\), ties, constraints and factor
binding are defined.

### OPEN I5. Resolved-memory LUT

Need typed keys and values.

Memory should represent **resolved experience**, not merely class prototypes.

A minimum contract is:

\[
MEMORY:
(\text{constraint state},\text{resolved state},\text{verification})
\]

with an exact rule for lookup, exclusion, construction, verification and
insertion.

v19 demonstrated one finite example of this cycle; it did not define the
general MPRC memory type.

---

## 6. Foundation closure gates before a full experiment

Scaling is blocked until these gates are green.

| Gate | Requirement | Status RC1 |
|---|---|---|
| G0 | \(Z_{256}\), cdist metric, polarity/origin constants | CLOSED |
| G1 | GEN7 full-cycle transport + exact inverse | CLOSED |
| G2 | 128x113 DATA+INFORMATION manifold arithmetic | CLOSED |
| G3 | BIND and wide MEASURE typed laws | CLOSED |
| G4 | identity-LUT REACT algebra + 7-round 113 dependency cone | CLOSED |
| G5 | 16-channel encoder codomain/integer closure | CLOSED |
| G6 | observation-to-structural packing map \(\Pi\) | **OPEN** |
| G7 | admissible ReactionLUT family/laws | **OPEN** |
| G8 | IDENTIFY state type and factor construction | **OPEN** |
| G9 | U/MOVE temporal state-transition law | **OPEN** |
| G10 | SELECT + resolved-memory contract | **OPEN** |
| G11 | one end-to-end finite theorem/gate using G0-G10 | BLOCKED |
| G12 | full dataset experiment | BLOCKED |

---

## 7. Immediate research order

Do **not** create v24 yet.

The next mathematical work is:

\[
\boxed{G6 \to G7 \to G8 \to G9 \to G10 \to G11}
\]

The highest-priority question is G6:

> What is the exact mathematical relationship between the owner's
> \(16\times113\times128\) wave observation state and the canonical
> \(128\times(98+15)\) structural DATA+INFORMATION manifold?

Until that is frozen, later attention experiments can produce numbers but do
not test one closed MPRC machine.

---

## 8. Scaling criterion

We are ready for the full experiment only when one diagram has no untyped
arrow:

\[
RGB/PCM/\text{other source}
\xrightarrow{OBSERVE}
\mathcal O
\xrightarrow{\Pi}
\mathcal S
\xrightarrow{IDENTIFY}
\mathcal F
\xrightarrow{BIND/REACT/MEASURE}
\mathcal E
\xrightarrow{MOVE/SELECT}
\mathcal R
\xrightarrow{MEMORY}
\mathcal R'.
\]

Every arrow must be either a DEFINITION with finite invariants or a THEOREM in
its stated domain.

That is the foundation freeze required before scale.
