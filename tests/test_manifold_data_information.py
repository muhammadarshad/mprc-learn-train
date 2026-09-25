"""Exact arithmetic regression for the corrected MPRC manifold construction."""

import math

D = 256
ROOT = math.isqrt(D)
assert ROOT * ROOT == D

N = ROOT - 1
GEN = ROOT // 2 - 1
H = D // 2

assert ROOT == 16
assert N == 15
assert GEN == 7
assert H == 128

WD = 2 * GEN * GEN
WI = N
W = WD + WI

assert WD == 98
assert WI == 15
assert W == 113
assert W == H - N

DATA = H * WD
INFO = H * WI
M = H * W

assert DATA == 12_544
assert INFO == 1_920
assert M == 14_464
assert DATA + INFO == M

# Equivalent counting views.
assert DATA == 49 * 256
assert DATA == 98 * 128
assert INFO == 15 * 128
assert M == 113 * 128

# 64 closure.
assert GEN * GEN + N == 64
assert 64 == D // 4

# Generator unit properties.
assert math.gcd(GEN, 64) == 1
assert math.gcd(GEN, 256) == 1
assert (GEN * 183) % 256 == 1

# Cache execution.
TILE_H = 64
assert H == 2 * TILE_H
assert TILE_H * W == 7_232
assert TILE_H * WD == 6_272
assert TILE_H * WI == 960
assert 2 * 7_232 == M
assert 2 * 6_272 == DATA
assert 2 * 960 == INFO

# MPRC U(1) completed-unit notation.
U1 = D
assert U1 == 256
assert (N + 1) ** 2 == U1

print("PASS: corrected MPRC DATA+INFORMATION manifold identities")
