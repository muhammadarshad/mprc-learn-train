# v25 — Orthogonal Factor Parallelogram Construction

**Status:** conditional algebra theorem + real-image empirical gate  
**Purpose:** test the main MPRC agenda directly: resolved factors should construct an unseen whole state.

---

## 1. Factor model

Let two orthogonal factors be

\[
s\in\{0,1\}
\]

and

\[
c\in\{0,1\}.
\]

Assume a state family in the additive \(Z_{256}\) module has the form

\[
\boxed{
X_{s,c}
=
B\oplus S_s\oplus C_c.
}
\]

Here:

- \(B\) is common state;
- \(S_s\) is the shape-factor displacement;
- \(C_c\) is the color-factor displacement;
- \(\oplus,\ominus\) are coordinate-wise \(Z_{256}\) addition/subtraction.

This is a **candidate factor-separability model** for real wave observations.

---

## 2. Parallelogram theorem

Under the factor model,

\[
X_{00}=B\oplus S_0\oplus C_0,
\]

\[
X_{01}=B\oplus S_0\oplus C_1,
\]

\[
X_{10}=B\oplus S_1\oplus C_0.
\]

Then

\[
X_{10}\oplus X_{01}\ominus X_{00}
\]

equals

\[
(B\oplus S_1\oplus C_0)
\oplus
(B\oplus S_0\oplus C_1)
\ominus
(B\oplus S_0\oplus C_0)
\]

and all common terms cancel in the additive group, leaving

\[
\boxed{
B\oplus S_1\oplus C_1
=
X_{11}.
}
\]

Therefore:

\[
\boxed{
X_{11}
=
X_{10}\oplus X_{01}\ominus X_{00}.
}
\]

This is an exact theorem **conditional on the additive orthogonal-factor model**.

By symmetry, any missing corner can be constructed from the other three.

---

## 3. Why this is the right constructive test

Whole-pattern memory has only the three resolved corners.

It does not contain the fourth combination.

The constructive system does not ask:

> "Have I seen the whole object?"

Instead it computes factor displacement.

For color:

\[
\Delta_C
=
X_{01}\ominus X_{00}.
\]

For shape:

\[
\Delta_S
=
X_{10}\ominus X_{00}.
\]

Then

\[
\boxed{
X_{11}^{*}
=
X_{00}\oplus\Delta_C\oplus\Delta_S.
}
\]

So resolved experience provides the displacements; BIND constructs the missing state.

---

## 4. Real 16-channel observation state

Use the owner-supplied fixed wave encoder:

\[
RGB
\to
\Psi
\in
Z_{256}^{16\times113\times128}.
\]

Apply the v24 channel-preserving frame interface:

\[
\Pi_T:
16\times113\times128
\to
16\times128\times113.
\]

No channel is dropped, averaged or manually labeled.

Thus the state used by v25 is

\[
\boxed{
\Psi^{S}
\in
(Z_{256}^{128\times113})^{16}.
}
\]

---

## 5. Real 2x2 factor cross

Use the same real Fruits-360 cross:

\[
\begin{array}{c|cc}
& orange & green\\
\hline
citrus & Orange & Lime\\
pepper & Orange\ Pepper & Green\ Pepper
\end{array}
\]

For each fold, one whole combination is absent from resolved memory.

The other three classes define the candidate parallelogram completion.

No held-out image participates in construction.

---

## 6. Resolved class state

Each visible class is represented by a medoid selected from its training observations.

The medoid is the resolved state minimizing total wide ring MEASURE to the other observations of the same class.

This is an empirical representative-selection rule, not a theorem.

---

## 7. Construct missing state

For a held-out corner \(X_{ab}\), let the other three resolved class states be the corresponding rectangle corners.

Construct

\[
\boxed{
\hat X_{ab}
=
X_{a,\bar b}
\oplus
X_{\bar a,b}
\ominus
X_{\bar a,\bar b}.
}
\]

All operations are coordinate-wise \(Z_{256}\) over all 16 structural manifolds.

No semantic shape/color vector is introduced.

---

## 8. Verification through the real MPRC attention path

For every query observation \(Q\), compare:

1. each of the three resolved whole-memory states;
2. the constructed missing state \(\hat X\).

For each of the 16 channels independently use the frozen path:

\[
TRANSPORT
\to
BIND
\to
REACT_{7}
\to
MEASURE.
\]

Sum the 16 wide integer energies.

Then

\[
\boxed{
SELECT=\arg\min E.
}
\]

The primary question is:

\[
\boxed{
\text{Does SELECT prefer the constructed unseen state over all remembered whole states?}
}
\]

This directly tests construction versus whole-pattern recall.

---

## 9. Required controls

Report:

- raw ring MEASURE before REACT;
- MPRC attention energy after 7 REACT rounds;
- visible-memory best energy;
- constructed-state energy;
- margin

\[
\boxed{
\Delta E
=
E_{\text{best visible}}
-
E_{\text{constructed}}.
}
\]

Positive margin means construction beats whole-memory recall.

Also report per-channel energies without selecting channels.

---

## 10. Status boundary

### THEOREM

If

\[
X_{s,c}=B\oplus S_s\oplus C_c,
\]

then the ring parallelogram completion is exact.

### EMPIRICAL

Whether real 16-channel wave observations of Orange/Lime/Pepper satisfy this additive factor model closely enough for the constructed state to win.

### NOT CLAIMED

- exact image synthesis;
- unsupervised discovery of semantic factors;
- proof that all vision factors are additive;
- final INFORMATION semantics;
- final U-observer law.

v25 is the direct constructive gate that was missing between the foundation mathematics and the larger experiment.
