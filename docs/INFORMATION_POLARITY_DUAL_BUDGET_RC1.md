# INFORMATION Polarity Dual-Budget Structure RC1

**Status:** exact arithmetic identities + implementation boundary  
**Date:** 26 September 2026

This note preserves the distinction between the half-domain INFORMATION budget

\[
15\times128=1,920
\]

and the full-domain INFORMATION budget

\[
15\times256=3,840.
\]

The second branch has not yet been exercised in the current implementation because
the current execution geometry does not instantiate a \(64\times256\) state.

---

## 1. Base constants

\[
D=256,
\qquad
H=\frac D2=128,
\qquad
N=\sqrt D-1=15.
\]

Also

\[
GEN^2+N
=
49+15
=
64.
\]

Therefore define the full arithmetic envelope

\[
\boxed{
E=(GEN^2+N)D=64\times256=16,384.
}
\]

This is an exact count identity.

---

## 2. Half-domain INFORMATION budget

The canonical polarity/hemisphere-aware INFORMATION count uses the half-domain height:

\[
\boxed{
I_{1/2}=NH=15\times128=1,920.
}
\]

Subtracting one half-domain INFORMATION field from the full envelope gives

\[
E-I_{1/2}
=
16,384-1,920
=
\boxed{14,464}.
\]

Therefore

\[
\boxed{
M=E-I_{1/2}=14,464.
}
\]

This is exactly the current canonical manifold count.

---

## 3. Full-domain INFORMATION budget

If the full 256-state INFORMATION extent is used,

\[
\boxed{
I_1=ND=15\times256=3,840.
}
\]

Since

\[
D=2H,
\]

we have

\[
\boxed{
I_1=2I_{1/2}.
}
\]

Subtracting the full INFORMATION field from the same envelope gives

\[
E-I_1
=
16,384-3,840
=
\boxed{12,544}.
\]

Thus

\[
\boxed{
DATA=E-I_1=12,544.
}
\]

And since

\[
GEN^2=49,
\]

\[
\boxed{
DATA=GEN^2D=49\times256=12,544.
}
\]

---

## 4. Exact two-step ladder

The three counts form an exact descending ladder:

\[
\boxed{
16,384
\;\xrightarrow{-1,920}\;
14,464
\;\xrightarrow{-1,920}\;
12,544.
}
\]

Equivalently,

\[
\boxed{
E
\to
M
\to
DATA
}
\]

with equal step size

\[
\boxed{
I_{1/2}=1,920.
}
\]

Because

\[
I_1=2I_{1/2},
\]

the direct relation is

\[
\boxed{
DATA=E-I_1.
}
\]

The canonical manifold relation remains

\[
\boxed{
M=DATA+I_{1/2}.
}
\]

---



## 4A. Dual structural layouts with invariant DATA

The owner's correction exposes two exact layouts.

### Full-domain / no-polarity layout

\[
\boxed{
64\times256=(49+15)\times256.
}
\]

Therefore

\[
DATA_{full}
=
49\times256
=
\boxed{12,544}
\]

and

\[
INFO_{full}
=
15\times256
=
\boxed{3,840}.
\]

Total:

\[
12,544+3,840
=
\boxed{16,384}.
\]

### Half-domain / polarity-aware layout

The current canonical construction is

\[
\boxed{
128\times113
=
128\times(98+15).
}
\]

Therefore

\[
DATA_{half}
=
98\times128
=
\boxed{12,544}
\]

and

\[
INFO_{half}
=
15\times128
=
\boxed{1,920}.
\]

Total:

\[
12,544+1,920
=
\boxed{14,464}.
\]

### DATA invariance

The DATA count is exactly preserved:

\[
\boxed{
49\times256
=
98\times128
=
12,544.
}
\]

Since

\[
98=2\cdot49
\]

and

\[
256=2\cdot128,
\]

there is an exact reindexing of the DATA rectangle obtained by splitting every
256-state row into two 128-state rows:

\[
\boxed{
R^{49\times256}
\cong
R^{98\times128}.
}
\]

This is a pure coordinate bijection and requires no information loss.

The INFORMATION blocks are not equal:

\[
\boxed{
15\times256
=
2(15\times128).
}
\]

The owner's interpretation is that the \(15\times128\) construction works in
both \(+\) and \(-\) polarity directions, whereas a construction not using
polarity requires the full \(15\times256\) extent.

The count relation is exact. The precise operational polarity map is still to
be frozen separately.


## 5. Compact algebraic form

Using

\[
E=(GEN^2+N)D,
\]

\[
I_{1/2}=N\frac D2,
\]

\[
I_1=ND,
\]

we obtain

\[
\boxed{
M
=
(GEN^2+N)D
-
N\frac D2
}
\]

and

\[
\boxed{
DATA
=
(GEN^2+N)D
-
ND
=
GEN^2D.
}
\]

For \(D=256, GEN=7, N=15\),

\[
M
=
64(256)-15(128)
=
14,464
\]

and

\[
DATA
=
64(256)-15(256)
=
49(256)
=
12,544.
\]

---

## 6. Polarity interpretation boundary

The owner's current distinction is:

- \(15\times128=1,920\): half-domain INFORMATION budget associated with the polarity/hemisphere construction.
- \(15\times256=3,840\): full-domain INFORMATION budget when the full 256-state extent is used.

The arithmetic identities are exact.

The semantic interpretation of how polarity is encoded or omitted in the full-domain branch is not expanded beyond the owner's stated construction in this note.

---

## 7. Implementation status

### Tested/current branch

The current structural implementation uses

\[
128\times113
=
14,464
\]

with

\[
98\ DATA + 15\ INFORMATION
\]

over the 128-row half-domain construction.

Therefore the

\[
15\times128=1,920
\]

branch is represented in current code and has passed the existing foundation gates.

### Untested branch

The full-domain budget

\[
15\times256=3,840
\]

has **not** yet been tested as an instantiated structural state.

In particular, the current execution code does not operate on

\[
\boxed{64\times256}.
\]

Therefore the identities

\[
64\times256=16,384
\]

and

\[
16,384-3,840=12,544
\]

are algebraically exact, but the \(64\times256\) execution/packing branch is currently:

\[
\boxed{\text{UNTESTED / NOT IMPLEMENTED}.}
\]

Do not report it as a code-survival result until a separate finite gate is built.

---

## 8. Status table

| Statement | Status |
|---|---|
| \(64\times256=16,384\) | EXACT IDENTITY |
| \(15\times128=1,920\) | EXACT IDENTITY / CURRENT BRANCH |
| \(15\times256=3,840\) | EXACT IDENTITY |
| \(3,840=2(1,920)\) | EXACT IDENTITY |
| \(16,384-1,920=14,464\) | EXACT IDENTITY |
| \(16,384-3,840=12,544\) | EXACT IDENTITY |
| \(12,544=49\times256=GEN^2D\) | EXACT IDENTITY |
| \(16,384\to14,464\to12,544\) in equal \(-1,920\) steps | EXACT IDENTITY |
| current \(15\times128\) structural branch | CODE-TESTED |
| \(15\times256\) / \(64\times256\) execution branch | UNTESTED |
