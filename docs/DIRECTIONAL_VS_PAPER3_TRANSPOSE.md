# Directional Spatial Transpose and Paper-3 T_k Are Distinct

**Status: proved incompatibility / no-conflation rule**

The corrected local vision context is

```
C, U1, U2, D1, D2, F1, F2, B1, B2
```

Ordinary image transpose acts as

```
C  -> C
U1 <-> B1
U2 <-> B2
D1 <-> F1
D2 <-> F2
```

Therefore its permutation cycle type on the nine local slots is

```
1^1 2^4
```

(one fixed slot and four two-cycles).

Paper 3 defines

```
T_k(a,p) = (p+k, a-k) mod 3
```

and proves that for every `k in Z3` it has three fixed local points and three
two-cycles.  Its cycle type is

```
1^3 2^3.
```

Conjugate permutations have the same cycle type. Hence there is no bijection
between the nine corrected directional slots and `Z3^2` that turns ordinary
spatial transpose into any one of the Paper-3 `T_k` maps.

## Consequence

The following must remain distinct:

1. **native byte spatial orientation transport**
   `16x7 <-> 7x16`, with directional arm transpose
   `U<->B, D<->F`;

2. **Paper-3 local algebraic Transpose**
   `T_k(a,p)` on the QH4 mixed-radix `Z3^2` coordinate.

Paper 3 still gives the exact address decomposition

```
m = 64*theta + 21*a + 7*p + sigma
z = 7*m mod 256
```

but spatial-image semantics must not be assigned to `(a,p)` merely to make the
numbers line up.

This removes one invalid route for closing the Arshad-ViT IDENTIFY/TRANSPORT seam.
