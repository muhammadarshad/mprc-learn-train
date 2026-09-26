# v26 — Relational IDENTIFY over the 16-Channel Wave State

**Status:** main-agenda constructive factor gate  
**Reason for v26:** v25 showed that applying the orthogonal-factor parallelogram directly to absolute image states is not valid for the real wave observations.

---

## 1. v25 diagnosis

v25 attempted

\[
\hat X_{ab}
=
X_{a,\bar b}
\oplus
X_{\bar a,b}
\ominus
X_{\bar a,\bar b}
\]

directly over the 16 full image-state manifolds.

The conditional algebra is correct if the observed states themselves satisfy

\[
X_{s,c}=B\oplus S_s\oplus C_c.
\]

The real fruit wave states do not satisfy that coordinate-wise model closely enough.

Therefore the correction is:

\[
\boxed{
\text{do not require factor orthogonality in absolute image coordinates.}
}
\]

Orthogonality belongs in the IDENTIFY/evidence state.

---

## 2. Observation

Use the owner-supplied fixed 16-channel encoder:

\[
QCMWaveEncoder:
RGB
\to
\Psi
\in
Z_{256}^{16\times113\times128}.
\]

Use the channel-preserving v24 frame interface:

\[
\Pi_T:
16\times113\times128
\to
16\times128\times113.
\]

No channel is dropped, averaged or preassigned a semantic factor.

---

## 3. IDENTIFY evidence vector

For observation \(A\) and resolved state \(B\), define the per-channel raw evidence

\[
\boxed{
E^{raw}(A,B)
=
(E_0,\ldots,E_{15})
}
\]

where

\[
E_k
=
\sum_{u}
d_{Z_{256}}(A_{k,u},B_{k,u}).
\]

Also define the per-channel MPRC attention evidence

\[
\boxed{
E^{attn}_k(A,B)
=
MEASURE(
REACT_7(BIND(B_k,A_k)),
A_k
).
}
\]

The full 16-dimensional vector is preserved.

This vector is the v26 IDENTIFY state.

---

## 4. Relational discovery of factor-support channels

In each held-out fold, the three visible combinations contain:

1. one pair with the **same shape but different color**;
2. one pair with the **same color but different shape**.

Let

\[
D^{shape}_k
\]

be the raw channel evidence between the same-shape/different-color resolved pair.

Let

\[
D^{color}_k
\]

be the raw channel evidence between the same-color/different-shape resolved pair.

Define:

\[
\boxed{
k\in C_{shape}
\iff
D^{shape}_k<D^{color}_k
}
\]

and

\[
\boxed{
k\in C_{color}
\iff
D^{color}_k<D^{shape}_k.
}
\]

If equal, the channel is **shared/undecided** and is not silently assigned.

There is:

- no manually declared "shape channel";
- no manually declared "color channel";
- no Top-K;
- no Softmax;
- no learned float weighting.

The factor-support sets are identified from resolved relations.

---

## 5. Resolved memory

Each visible whole class has one resolved medoid state selected using all 16 raw ring channel energies.

The whole class combination itself remains a memory object.

From those same resolved observations, v26 also constructs two factor-memory views:

\[
MEM_{shape}
\]

and

\[
MEM_{color}.
\]

These are not new image encoders. They are views of the same 16-channel state under the relationally discovered channel sets.

---

## 6. Independent factor SELECT

For a query \(Q\), evaluate candidate shape values using only

\[
C_{shape}
\]

and candidate color values using only

\[
C_{color}.
\]

For either raw or MPRC-attention evidence:

\[
score_{shape}(s)
=
\min_{m:\ shape(m)=s}
\sum_{k\in C_{shape}}E_k(Q,m)
\]

and

\[
score_{color}(c)
=
\min_{m:\ color(m)=c}
\sum_{k\in C_{color}}E_k(Q,m).
\]

Then

\[
\boxed{
\hat s=\arg\min_s score_{shape}(s)
}
\]

and

\[
\boxed{
\hat c=\arg\min_c score_{color}(c).
}
\]

Finally construct the response:

\[
\boxed{
\hat y=BIND(\hat s,\hat c)
}
\]

where BIND here means composition of two independently selected resolved
factor identities, not pixel arithmetic.

The pair

\[
(\hat s,\hat c)
\]

may never have existed in whole-object memory.

---

## 7. Whole-memory control

A separate control uses all 16 channels together and may select only one of the
three visible whole combinations.

Therefore when the fourth combination is held out, whole-memory recall cannot
construct it.

The useful comparison is not "memory bad, construction good".

Memory supplies the resolved factor states.

The test is:

\[
\boxed{
\text{Can relational IDENTIFY reuse those resolved factors to construct the absent combination?}
}
\]

---

## 8. Required outputs

For both raw and 7-round attention evidence report:

- shape SELECT accuracy;
- color SELECT accuracy;
- constructed pair accuracy;
- per-fold factor-support channel sets;
- undecided/shared channels;
- whole-memory selected class;
- actual photo audit.

The primary metric is:

\[
\boxed{
\text{constructed held-out pair correct}
}
\]

not synthetic-image similarity.

---

## 9. Claim boundary

v26 is a controlled real-image orthogonal-factor construction gate.

Factor labels define the controlled two-factor problem.

The channel allocation itself is not manually specified; it is identified from
visible relations.

A positive result would establish evidence-level composition for this controlled
cross. It would not prove unsupervised discovery of arbitrary semantic factors.
