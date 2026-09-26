# MPRC Coprime Manifold Configurations RC1

**Status:** configuration rule + discovered configurations + exact CRT traversal theorem  
**Date:** 26 September 2026

This note freezes an important distinction:

\[
\boxed{\text{arithmetic budget} \neq \text{valid MPRC configuration}.}
\]

A valid MPRC rectangular configuration used in the current framework has the form

\[
\boxed{
H\times W
}
\]

with

\[
\boxed{
H=2^n,\qquad \gcd(H,W)=1.
}
\]

Because \(H\) is a power of two, \(W\) must be odd.

The two configurations currently used/discovered in MPRC are

\[
\boxed{
64\times157
}
\]

and

\[
\boxed{
128\times113.
}
\]

These dimensions are recorded as **DISCOVERED / FROZEN CONFIGURATIONS**.

The coprimality rule does not, by itself, uniquely derive either 157 or 113.
Do not rewrite their history as if those widths were forced merely by the
condition \(H=2^n,\gcd(H,W)=1\).

---

## 1. Current discovered configurations

### Configuration A

\[
\boxed{
C_{64}=64\times157.
}
\]

Checks:

\[
64=2^6
\]

and

\[
\gcd(64,157)=1.
\]

Total states:

\[
\boxed{
64\cdot157=10,048.
}
\]

### Configuration B

\[
\boxed{
C_{128}=128\times113.
}
\]

Checks:

\[
128=2^7
\]

and

\[
\gcd(128,113)=1.
\]

Total states:

\[
\boxed{
128\cdot113=14,464.
}
\]

---

## 2. Why coprimality matters — CRT theorem

For positive integers \(H,W\),

\[
\mathbb Z_H\times\mathbb Z_W
\]

is cyclic iff

\[
\gcd(H,W)=1.
\]

By the Chinese Remainder Theorem,

\[
\boxed{
\mathbb Z_{HW}
\cong
\mathbb Z_H\times\mathbb Z_W
}
\]

when \(H,W\) are coprime.

The coordinate map

\[
\phi(t)
=
(t\bmod H,\ t\bmod W)
\]

is then a bijection over exactly \(HW\) states.

So a single scalar traversal index can address the entire rectangular
configuration without splitting into shorter repeated coordinate cycles.

This is an exact theorem.

---

## 3. GEN7 full-cycle traversal on both discovered configurations

Let

\[
g=GEN=7.
\]

A generator traversal is

\[
\phi_g(t)
=
(gt\bmod H,\ gt\bmod W).
\]

If

\[
\gcd(g,H)=\gcd(g,W)=1,
\]

then multiplication by \(g\) is invertible in each coordinate.

Combined with

\[
\gcd(H,W)=1,
\]

the map traverses all \(HW\) coordinate pairs exactly once as
\(t=0,\ldots,HW-1\).

### \(64\times157\)

\[
\gcd(7,64)=1,
\qquad
\gcd(7,157)=1.
\]

Therefore GEN7 gives a complete \(10,048\)-state traversal.

### \(128\times113\)

\[
\gcd(7,128)=1,
\qquad
\gcd(7,113)=1.
\]

Therefore GEN7 gives a complete \(14,464\)-state traversal.

---

## 4. 64x256 is NOT a valid configuration

The arithmetic product

\[
64\times256=16,384
\]

is useful as a count/envelope.

But

\[
\gcd(64,256)=64.
\]

Therefore

\[
\boxed{
64\times256
}
\]

does not satisfy the current MPRC configuration rule.

Its joint coordinate traversal decomposes into repeated shorter cycles rather
than one CRT cycle of length \(64\cdot256\).

Hence it must be labeled:

\[
\boxed{
\text{ARITHMETIC ENVELOPE / COUNT, NOT A CURRENT MPRC CONFIGURATION}.
}
\]

---

## 5. INFORMATION budgets must not be confused with configurations

The exact counts

\[
15\times128=1,920
\]

and

\[
15\times256=3,840
\]

are INFORMATION extents/budgets.

They do not independently define a valid rectangular MPRC configuration.

In particular,

\[
15\times256=3,840
\]

does **not** imply that the system should run as

\[
64\times256.
\]

The current \(128\times113\) configuration uses the half-domain
\(15\times128\) construction.

The full-domain \(15\times256\) INFORMATION budget has not yet been mapped to a
tested valid coprime MPRC configuration.

That interface remains OPEN.

---

## 6. Discovered, not forced

The two current configurations

\[
64\times157
\]

and

\[
128\times113
\]

should be described as:

\[
\boxed{
\text{discovered configurations satisfying the MPRC coprimality rule}.
}
\]

Do not say:

- 157 was forced solely by the power-of-two rule;
- 113 was forced solely by coprimality;
- every \(2^n\times\text{odd}\) rectangle is an accepted MPRC configuration.

There are infinitely many odd widths coprime to a given \(2^n\).

The rule provides an admissibility condition. The current widths are the
specific configurations discovered and retained by the framework.

---

## 7. Status table

| Statement | Status |
|---|---|
| current configuration height is \(2^n\) | FROZEN CONFIGURATION RULE |
| \(\gcd(H,W)=1\) | FROZEN CONFIGURATION RULE |
| \(64\times157\) | DISCOVERED / CURRENT CONFIGURATION |
| \(128\times113\) | DISCOVERED / CURRENT CONFIGURATION |
| \(\gcd(64,157)=1\) | EXACT |
| \(\gcd(128,113)=1\) | EXACT |
| CRT gives \(\mathbb Z_{HW}\cong\mathbb Z_H\times\mathbb Z_W\) when coprime | THEOREM |
| GEN7 traverses all states in both current configurations | THEOREM / FINITE GATE |
| \(64\times256\) | ARITHMETIC COUNT ONLY |
| \(64\times256\) as current MPRC configuration | REJECTED |
| \(15\times256=3,840\) INFORMATION budget | EXACT COUNT / UNMAPPED |
| mapping \(3,840\) to a valid current configuration | OPEN |
