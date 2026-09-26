# v28 — Typed QH4 + Directional ADI IDENTIFY

**Status: exact interface theorem. No training.**

The corrected local directional state is

```
A = (C,U1,U2,D1,D2,F1,F2,B1,B2) in Z256^9.
```

Directional ADI is the exact bijection

```
A
  -> D = (Lambda,delta1,...,delta8)
  in Z256^9.
```

For one byte `v in Z256`, QH4 gives an exact tagged partition:

```
v in {0,64,128,192}
    -> VacuumTag(v)

otherwise
    -> ActiveTag(theta,a,p,sigma)
```

where the active tag is the exact Paper-3 mixed-radix inverse and reconstructs
the byte through

```
m = 64*theta + 21*a + 7*p + sigma
v = 7*m mod 256.
```

Therefore the component-wise composition

```
IDENTIFY(A)
  = (ADI9(A), QH4Tag(ADI9(A)))
```

is lossless.

## What this closes

It closes the **type composition** in

```
IDENTIFY = QH4 + ADI-9
```

without forcing the corrected directional slot names themselves to be the
Paper-3 `(a,p)` coordinate.

That distinction is required because the directional spatial transpose and
Paper-3 `T_k` have already been proved non-conjugate.

## What remains open

v28 does **not** define

```
IDENTIFY -> routing/state transition
```

or how a tagged ADI relation changes/selects the manifold/query entering BIND.

It also does not define the learned ReactionLUT.

Those are still pre-training seams.
