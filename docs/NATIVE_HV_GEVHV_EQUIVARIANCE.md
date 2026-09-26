# Native H/V Transpose Equivariance of the GEVHV Core

**Status: exact theorem + finite code gate**

For one native byte block

```
A in Z256^(16x7)
Q in Z256^(16x7)
```

let `T` be ordinary rectangular transpose into `7x16`.

Coordinate-wise vector BIND satisfies

```
T(A + Q) = T(A) + T(Q) mod 256.
```

The five-site REACT pre-LUT sum is symmetric under exchange of the two spatial
axes. Therefore, for any 256-entry pointwise LUT `L`,

```
T(REACT_L(A)) = REACT_L(T(A)).
```

The same holds for any number of repeated REACT rounds.

MEASURE is a sum of coordinate-wise circular distances, so transpose only
permutes terms:

```
MEASURE(A,Q) = MEASURE(T(A),T(Q)).
```

Therefore the entire core

```
BIND -> REACT -> MEASURE
```

is equivariant/invariant under the native

```
16x7 <-> 7x16
```

spatial orientation transport, provided state and query are transported together.

## Consequence

The H/V byte transpose is an exact coordinate phase transport; it does not need
a new learned parameter or a second reaction law.

This does **not** determine how QH4/directional IDENTIFY chooses an orientation
phase, nor does it make the spatial transpose equal to Paper-3 `T_k`.
