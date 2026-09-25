# Global Attention Probe — Coding/Research Prompt

## Objective

Build the **Global Attention (GA) probe** for MPRC using the corrected local/global model and Arshad's Transpose.

Do **not** interpret `128x113` as image geometry, a receptive field, or a pixel neighborhood.

The probe must answer one question:

> Can a query expressed in the local `Z256` state space retrieve the correct globally assigned occurrences across `HV*N = 14,464*N` positions, and then let BIND -> REACT -> MEASURE rank those retrieved occurrences without a dense attention matrix?

No classifier accuracy claim is part of this probe.

---

## Frozen arithmetic

```text
D       = 256
LOCAL   = Z256
4x4     = 16
16^2    = 256 local ordered relation/state count

HV      = 128*113 = 14,464 states per manifold/sample
GLOBAL  = HV*N = 14,464*N globally unique positions

GEN     = 7
GEN_INV = 183 mod 256
```

For sample/manifold `n` and manifold position `h`:

[
g=ncdot14,464+h
]

with

[
0le h<14,464,qquad 0le n<N.
]

Every occurrence has exactly one globally unique `g`.

Its local state is

[
p=X(g)in Z_{256}.
]

Two occurrences may have the same `p`; they must still remain distinct because their global positions differ.

---

## Arshad's Transpose

Represent the forward state as

[
X:G_N	o Z_{256}.
]

Build the transpose/inverted occurrence map

[
X^T[p]={g:X(g)=p}.
]

Implement this as a sparse CSR-style index:

```text
offsets[257]
positions[HV*N]
```

NOT as a dense

```text
256 x (HV*N)
```

matrix.

Required properties:

1. The 256 buckets are disjoint.
2. Their union is every global position exactly once.
3. `sum(bucket_size[p]) = HV*N`.
4. Transpose -> reconstruction recovers `X` exactly.
5. Bucket occupancy is data-dependent. Never assume equal bucket sizes.
6. Repeated local states retain all distinct global occurrences.
7. Increasing `N` enlarges only the GLOBAL domain; LOCAL remains exactly 256.

The gate `scripts/local_global_transpose_survival_gate.py` must pass before GA code is evaluated.

---

## Local query navigation

A query begins in the local state domain:

[
p_qin Z_{256}.
]

Generator-7 navigation is

[
p_t=(p_q+7t)mod256.
]

Because

[
gcd(7,256)=1,
]

the full orbit contains all 256 local states, and

[
7^{-1}=183pmod{256}.
]

Do **not** automatically probe all 256 states during normal attention. The GA probe must expose the local relation policy as an explicit interface.

Create something equivalent to:

```python
class LocalRelationPolicy:
    def states(self, query_local_state: int, query_context) -> list[int]:
        ...
```

For the initial probe provide at least:

- `ExactStatePolicy`: only `p_q`
- `GeneratorPrefixPolicy(k)`: first `k` states of the generator-7 orbit
- `ExplicitRelationPolicy`: caller supplies exact local related states

These are probe controls. Do not claim one is the learned attention rule.

---

## Global candidate collection

For each local state returned by the policy:

```text
local state p
    -> X^T[p]
    -> list of global occurrence IDs g
```

Union the global IDs without losing identity.

Return candidates with both coordinates:

```text
(local_state p, global_position g)
```

and decoded global identity:

```text
sample = g // 14464
site   = g % 14464
```

A local match is not a global match until `g` is retained.

---

## Attention probe

Only after candidate retrieval, run the existing MPRC evidence path on candidate states:

[
oxed{	ext{BIND}ightarrow	ext{REACT}ightarrow	ext{MEASURE}}
]

The probe must compare at least:

### A. Self retrieval
A query occurrence must retrieve itself under exact-state transpose lookup.

Report:

```text
self_retrieved / queries
```

Required mathematical expectation: exact-state lookup gives 100% self inclusion.

### B. Duplicate-local test
Create many global occurrences with identical local state `p`.

The transpose must retrieve **all** their distinct global positions.

### C. Global-position discrimination
Construct pairs with identical local state but different global information/context.

The attention path must preserve them as separate candidates. If MEASURE cannot distinguish them when information differs, report that directly.

### D. Generator relation test
For a query `p`, probe controlled generator prefixes:

```text
K = 1, 2, 4, 7, 14, 16, 32, 64
```

Report:

- local states visited
- global candidates retrieved
- duplicates removed
- candidate recall
- MEASURE ordering

Do not choose a K from accuracy in this probe.

### E. Scaling
Run synthetic/global-index tests for:

```text
N = 1, 2, 7, 16, 64
```

Report:

- total global states = `14464*N`
- transpose build time
- offsets bytes
- positions bytes
- query lookup time
- candidate count
- dense-equivalent bytes avoided

---

## Critical distinction

The architecture is:

[
oxed{
	ext{LOCAL }Z_{256}
longleftrightarrow
	ext{GLOBAL }14,464N
}
]

not

[
128	imes113	ext{ pixel attention}.
]

The transpose is what joins them.

The intended flow is:

[
oxed{
q
ightarrow
	ext{IDENTIFY local }p_q
ightarrow
	ext{local relation policy}
ightarrow
X^T
ightarrow
	ext{global candidates}
ightarrow
	ext{BIND}
ightarrow
	ext{REACT}
ightarrow
	ext{MEASURE}
}
]

Do not insert softmax, Q/K/V, Euclidean cosine attention, FFT features, or a dense `N x N` attention matrix.

---

## SU(16) / local-state note

For this probe retain only the cardinality fact:

[
4	imes4=16,qquad16^2=256.
]

The MPRC research interpretation associates the completed 256-state local unit with the SU(16)/completion construction. Do not turn that interpretation into a standard Lie-group identity inside the code or proof.

---

## Required machine-readable report

Emit JSON with:

```json
{
  "local_domain": 256,
  "hv": 14464,
  "N": "...",
  "global_domain": "14464*N",
  "transpose": {
    "roundtrip": true,
    "positions_preserved": ".../... ",
    "bucket_min": 0,
    "bucket_max": 0,
    "offset_count": 257
  },
  "self_retrieval": {
    "numerator": 0,
    "denominator": 0
  },
  "generator_prefix": {},
  "scaling": {},
  "measure": {},
  "claim_boundary": "..."
}
```

Never report only percentages; always report numerator/denominator.

---

## Stop conditions

The GA probe fails if any of the following occurs:

- a global occurrence is lost;
- one global occurrence appears in multiple local buckets;
- transpose reconstruction differs from the forward mapping;
- repeated local states are collapsed into one global item;
- global identity is replaced by local state;
- `128x113` is reintroduced as image/pixel adjacency;
- a dense attention matrix is required;
- softmax or conventional QKV is inserted;
- generator 7 is replaced to improve a result.

If the transpose gate and self-retrieval tests pass but MEASURE cannot distinguish globally different occurrences with the same local state, report that as the next open attention problem rather than modifying the indexing geometry.
