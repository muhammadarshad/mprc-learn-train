# Canonical MPRC Data + Information Manifold

**Status:** corrected/frozen construction note  
**Base domain:** `D = 256`  
**Important:** this document defines a byte/state manifold, **not an image or pixel geometry**.

The canonical manifold is

[
oxed{M = DATA + INFORMATION = 12{,}544 + 1{,}920 = 14{,}464}
]

with

[
oxed{H=128,qquad W=113,qquad M=H,W.}
]

The 12,544 bytes are **DATA**. The 1,920 bytes are **INFORMATION**. Their source modality and external configuration are not part of this construction.

---

## 1. Base construction

Start from

[
oxed{D=256.}
]

Then

[
sqrt D = 16.
]

Define

[
oxed{N=sqrt D-1=15}
]

and

[
oxed{GEN=rac{sqrt D}{2}-1=7.}
]

The half-domain height is

[
oxed{H=rac D2=128.}
]

The DATA width is

[
oxed{W_D=2,GEN^2=2(7^2)=98.}
]

The INFORMATION width is

[
oxed{W_I=N=15.}
]

Therefore

[
oxed{W=W_D+W_I=98+15=113.}
]

Equivalently,

[
oxed{W=rac D2-sqrt D+1=113.}
]

---

## 2. DATA and INFORMATION byte counts

### DATA

[
oxed{DATA=H,W_D}
]

so

[
DATA
=128(98)
=oxed{12{,}544}.
]

Equivalent exact factorizations are

[
oxed{12{,}544=128cdot7cdot14}
]

and

[
oxed{12{,}544=49cdot256.}
]

Thus the same DATA region may be counted as either

[
oxed{49	ext{ full }Z_{256}	ext{ units}}
]

or

[
oxed{98	ext{ half-domain }128	ext{ units}.}
]

### INFORMATION

[
oxed{INFORMATION=H,N}
]

so

[
INFORMATION
=128(15)
=oxed{1{,}920}.
]

The 1,920 bytes are structural INFORMATION. They are **not padding** and are not derived from a pixel layout.

### Whole manifold

[
oxed{
M
=
H(2GEN^2+N)
}
]

therefore

[
M
=
128(98+15)
=
128(113)
=
oxed{14{,}464}.
]

Entirely in terms of (D),

[
oxed{
M
=
rac D2
left(
rac D2-sqrt D+1
ight).
}
]

For (D=256),

[
M=128(113)=14{,}464.
]

---

## 3. Equivalent 49/98/113 decomposition

The construction also reads

[
oxed{
49(256)+15(128)=14{,}464.
}
]

Since

[
49(256)=98(128),
]

we obtain

[
oxed{
98(128)+15(128)=113(128).
}
]

Hence

[
oxed{113=98+15}
]

is a structural DATA + INFORMATION width, not a receptive-field size or pixel count.

---

## 4. The 64 identity and why GEN = 7 closes the manifold

Because

[
GEN^2=7^2=49
]

and

[
N=15,
]

we have

[
oxed{GEN^2+N=49+15=64=rac D4.}
]

Therefore

[
GEN^2
=
rac D4-N
=
rac D4-sqrt D+1
=
left(rac{sqrt D}{2}-1ight)^2.
]

Thus

[
oxed{GEN=rac{sqrt D}{2}-1=7.}
]

The full-cycle arithmetic also survives independently:

[
gcd(7,64)=1,
qquad
gcd(7,256)=1,
]

and

[
oxed{7^{-1}equiv183pmod{256}.}
]

The exhaustive (Z_{64}) generator sweep in this repository found that (g=7) is the unique full-cycle candidate with zero error on all three independently derived closure conditions

[
g^2=49,
qquad
2g^2=98,
qquad
2g^2+15=113.
]

This structural result is separate from CPU traversal speed.

---

## 5. Cache execution: two 64-row passes

The manifold height is

[
128=2cdot64.
]

Do not process the full 128-state height as one cache pass. Use two cache-resident slabs:

[
oxed{
128	imes113
=
2(64	imes113).
}
]

Each slab is

[
64	imes113
=
oxed{7{,}232	ext{ bytes}}.
]

Per slab:

[
64	imes98
=
oxed{6{,}272	ext{ DATA bytes}}
]

and

[
64	imes15
=
oxed{960	ext{ INFORMATION bytes}}.
]

Therefore

[
oxed{
7{,}232=6{,}272+960.
}
]

Two passes reconstruct the full byte budget:

[
2(6{,}272)=12{,}544
]

and

[
2(960)=1{,}920.
]

The execution identity is therefore

[
oxed{
M
=
2cdot64
left[
(7cdot14)_{	ext{DATA}}
+
15_{	ext{INFORMATION}}
ight].
}
]

A direct nested traversal is:

```cpp
for (int slab = 0; slab < 2; ++slab) {
    for (int h = 0; h < 64; ++h) {

        // DATA width: 98 = 7 * 14
        for (int g = 0; g < 7; ++g) {
            for (int s = 0; s < 14; ++s) {
                const int w = g * 14 + s;   // 0..97
                // DATA state
            }
        }

        // INFORMATION width: 15
        for (int n = 0; n < 15; ++n) {
            const int w = 98 + n;           // 98..112
            // INFORMATION state
        }
    }
}
```

The physical slab can remain contiguous in L1. `GEN=7` is the logical/navigation generator; it does not require a cache-hostile physical stride.

---

## 6. Full state: U(1) is the completed 256-state unit

In the MPRC construction used here,

[
oxed{U(1)=256}
]

is one completed unit. Do **not** reduce (U(1)) to the (+1) in (15+1).

The construction is written in MPRC notation as

[
oxed{U(1)=SU(4),SO(1)}
]

with the internal count

[
4^2-1=15
]

closed to

[
15+1=16
]

and then completed as

[
oxed{16^2=256=U(1).}
]

Thus

[
oxed{
N=sqrt{U(1)}-1=15
}
]

and

[
oxed{
GEN=rac{sqrt{U(1)}}2-1=7.
}
]

This is an **MPRC structural notation/construction**, not a claim that the displayed expression is a standard Lie-group identity.

The essential hierarchy is

[
oxed{
15+1
ightarrow
16
ightarrow
16^2
ightarrow
256=U(1).
}
]

The full state therefore retains the completed (U(1)=256) unit; the 15 INFORMATION lanes are not a replacement for it.

---

## 7. Hemisphere pair must remain separate

Let

[
oxed{P=a+b}
]

and

[
oxed{Q=c+d.}
]

These must remain separate because they may occupy different hemispheres of the (Z_{256}) state around the origin (128):

- values below (128) and
- values above (128)

carry different polarity/hemisphere information.

Therefore the primary relational state is

[
oxed{
W^{(2)}=(P,Q)=(a+b,;c+d).
}
]

Do not collapse it immediately to (PQ), (P^2Q^2), or another scalar.

Squaring removes sign/polarity information:

[
(+x)^2=(-x)^2.
]

Hence scalar quantities such as

[
oxed{I_1=P^2Q^2}
]

or

[
oxed{I_2=(P^2Q^2)^2}
]

are **derived invariants**, not the complete MPRC state.

The order is

[
oxed{
(P,Q)
ightarrow
	ext{hemisphere/phase relation}
ightarrow
	ext{derived invariant}.
}
]

---

## 8. Tower levels must not be conflated

Two equations that have appeared in the research record are different constraints.

If

[
P^2Q^2=256,
]

then

[
I_1=256
]

and

[
I_2=256^2=65{,}536.
]

If instead

[
(P^2Q^2)^2=256,
]

then

[
P^2Q^2=16
]

and therefore

[
|PQ|=4
]

over the reals.

Do not silently identify these two tower levels.

---

## 9. Canonical compact form

The corrected manifold construction is

[
oxed{
egin{aligned}
U(1)&=D=256,\
sqrt D&=16,\
N&=sqrt D-1=15,\
GEN&=rac{sqrt D}{2}-1=7,\
H&=rac D2=128,\
W_D&=2GEN^2=98,\
W_I&=N=15,\
W&=W_D+W_I=113,\
DATA&=H,W_D=12{,}544,\
INFORMATION&=H,W_I=1{,}920,\
M&=H,W=14{,}464.
end{aligned}
}
]

and cache execution is

[
oxed{
M
=
2cdot64
left[
(7cdot14)_{	ext{DATA}}
+
15_{	ext{INFORMATION}}
ight].
}
]

The primary two-hemisphere relational state remains

[
oxed{
W^{(2)}=(a+b,;c+d)
}
]

before any scalar invariant is formed.

---

## 10. Deprecated interpretation

The earlier interpretation of this file as

[
(112+16)	imes(112+1)
]

with fixed image-channel rows and a polarity column is **deprecated**.

The canonical statement is now:

[
oxed{
14{,}464
=
12{,}544_{	ext{DATA}}
+
1{,}920_{	ext{INFORMATION}}.
}
]

No pixel geometry, image resolution, or modality-specific layout is implied by those byte counts.
