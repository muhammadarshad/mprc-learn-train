# 112-State Block / Local Factorization

The canonical 112x112 payload has a stronger exact decomposition than a generic square image.

## Horizontal phase

Use native 16x7 byte structures.

- block grid: 7x16 = 112 blocks
- local block shape: 16x7 = 112 positions

For pixel `(r,c)`:

```text
r = 16*block_row + local_row
c =  7*block_col + local_col

block_H = 16*block_row + block_col
local_H = 7*local_row + local_col
```

Thus:

```text
112x112 pixels <-> 112 block states x 112 local-position states
```

bijectively.

## Vertical / transposed phase

After 16x7 -> 7x16 transport:

```text
block_V = 7*(block_H mod 16) + floor(block_H/16)
local_V = 16*(local_H mod 7) + floor(local_H/7)
```

Both maps are permutations of exactly 112 states.

## Consequence for structural storage

Given the owner's frozen interpretation of the 128x113 manifold:

```text
128 = 112 pixels + 16 channel rows
113 = 112 pixels + 1 polarity column
```

the 16x112 channel region is exactly large enough to hold **one local-position state per channel per native block**:

```text
M[channel][block] in {0,...,111}
```

without storing or replacing the pixels themselves.

The exact transport law is:

```text
M'_c[block_H_to_V(b)] = local_H_to_V(M_c[b])
```

The 112x1 polarity region can transport one opaque polarity state per native block by the same block permutation:

```text
P'[block_H_to_V(b)] = P[b]
```

Spatial transpose does not by itself change polarity.

### Status boundary

The block/local factorization and the two 112-state permutations are exact arithmetic identities and are exhaustively verified.

The statement that a channel row should contain a selected local position per block is the current **metadata-codec hypothesis**, motivated by the owner's statement that structural storage is where position goes. The selector that determines which local position each channel stores is intentionally not invented here. It must come from the MPRC IDENTIFY/SELECT rule or be established empirically before freezing.
