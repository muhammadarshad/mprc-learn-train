#!/usr/bin/env python3
"""
Exhaustive 64-state generator sweep for the 128x113 DATA+INFORMATION manifold.

NO TRAINING. NO LABELS. NO WEIGHTED "BEST" SCORE.

Independent frozen arithmetic:
  DATA_BYTES = 12544
  INFO_BYTES = 1920
  MANIFOLD_BYTES = 14464
  H = 128
  W = 113
  CACHE_TILE_H = 64

Derived, without naming any candidate generator:
  data_full_ring_units = 12544 / 256 = 49
  data_half_ring_lanes  = 12544 / 128 = 98
  info_half_ring_lanes  = 1920 / 128 = 15
  98 + 15 = 113

We sweep every additive stride g=1..63 on Z64:
    q_t = q_0 + g*t mod 64.

Exact metrics are reported individually. No arbitrary weighted score is used.
"""
from pathlib import Path
import csv, json, math

DATA_BYTES=12544
INFO_BYTES=1920
MANIFOLD_BYTES=14464
H=128
W=113
TILE_H=64
RING=256
HALF=128
CACHELINE=64

assert DATA_BYTES+INFO_BYTES==MANIFOLD_BYTES
assert H*W==MANIFOLD_BYTES
assert DATA_BYTES % RING == 0
assert DATA_BYTES % HALF == 0
assert INFO_BYTES % HALF == 0

DATA_FULL=DATA_BYTES//RING      # 49
DATA_HALF=DATA_BYTES//HALF      # 98
INFO_HALF=INFO_BYTES//HALF      # 15
assert DATA_HALF+INFO_HALF==W
assert TILE_H*W==7232
assert (TILE_H*W)%CACHELINE==0
TILE_CACHELINES=(TILE_H*W)//CACHELINE   # 113

def orbit(g,n=64,q0=0):
    return [((q0+g*t)&63) for t in range(n)]

def period(g):
    return 64//math.gcd(g,64)

def circ64(a,b):
    d=(a-b)&63
    return min(d,(-d)&63)

def prefix_min_sep(g,n):
    s=orbit(g,n)
    return min(circ64(a,b) for i,a in enumerate(s) for b in s[i+1:])

def quarter_counts(g,n):
    c=[0,0,0,0]
    for q in orbit(g,n):
        c[q//16]+=1
    return c

def quarter_l1_discrepancy(g,n):
    c=quarter_counts(g,n)
    # Multiply by 4 to stay integer: sum |4*c_i - n|.
    return sum(abs(4*x-n) for x in c)

def lines_for_row(r):
    lo=r*W
    hi=lo+W-1
    return set(range(lo//CACHELINE,hi//CACHELINE+1))

def prefix_cache_lines(g,n):
    lines=set()
    for r in orbit(g,n):
        lines.update(lines_for_row(r))
    return len(lines)

rows=[]
for g in range(1,64):
    p=period(g)
    full=(p==64)
    unit256=(math.gcd(g,256)==1)
    inv256=pow(g,-1,256) if unit256 else None
    o=orbit(g)
    unique=len(set(o[:p]))
    row={
        "g":g,
        "gcd64":math.gcd(g,64),
        "period64":p,
        "full_cycle64":full,
        "unique_before_repeat":unique,
        "unit_z256":unit256,
        "inverse_mod256":inv256,
        "g_squared":g*g,
        "payload_full_ring_units":DATA_FULL,
        "square_closure_error":abs(g*g-DATA_FULL),
        "payload_half_ring_lanes":DATA_HALF,
        "double_square_closure_error":abs(2*g*g-DATA_HALF),
        "info_half_ring_lanes":INFO_HALF,
        "manifold_width":W,
        "width_closure_error":abs((2*g*g+INFO_HALF)-W),
        "popcount_g":g.bit_count(),
        "first7_min_circular_separation":prefix_min_sep(g,7) if p>=7 else 0,
        "first14_min_circular_separation":prefix_min_sep(g,14) if p>=14 else 0,
        "first7_quarter_counts":quarter_counts(g,7),
        "first14_quarter_counts":quarter_counts(g,14),
        "first7_quarter_l1_discrepancy_x4":quarter_l1_discrepancy(g,7),
        "first14_quarter_l1_discrepancy_x4":quarter_l1_discrepancy(g,14),
        "first7_cache_lines_touched":prefix_cache_lines(g,7),
        "first14_cache_lines_touched":prefix_cache_lines(g,14),
        "full_pass_cache_lines_touched":prefix_cache_lines(g,64),
    }
    rows.append(row)

full=[r for r in rows if r["full_cycle64"]]
assert len(full)==32
assert all(r["unit_z256"] for r in full)
assert all(r["full_pass_cache_lines_touched"]==TILE_CACHELINES for r in full)

def winners_min(key,domain=full):
    m=min(r[key] for r in domain)
    return m,[r["g"] for r in domain if r[key]==m]

def winners_max(key,domain=full):
    m=max(r[key] for r in domain)
    return m,[r["g"] for r in domain if r[key]==m]

summary={
    "constants":{
        "data_bytes":DATA_BYTES,
        "information_bytes":INFO_BYTES,
        "manifold_bytes":MANIFOLD_BYTES,
        "H":H,"W":W,"tile_H":TILE_H,
        "tile_bytes":TILE_H*W,
        "tile_cache_lines_64B":TILE_CACHELINES,
        "data_full_ring_units":DATA_FULL,
        "data_half_ring_lanes":DATA_HALF,
        "info_half_ring_lanes":INFO_HALF,
    },
    "sweep":{
        "g_values":"1..63",
        "candidate_count":63,
        "full_cycle_generators":[r["g"] for r in full],
        "full_cycle_count":len(full),
    },
    "structural_closure":{
        "square_error_min":winners_min("square_closure_error"),
        "double_square_error_min":winners_min("double_square_closure_error"),
        "width_error_min":winners_min("width_closure_error"),
    },
    "generic_sampling_diagnostics":{
        "first7_min_separation_max":winners_max("first7_min_circular_separation"),
        "first14_min_separation_max":winners_max("first14_min_circular_separation"),
        "first7_quarter_discrepancy_min":winners_min("first7_quarter_l1_discrepancy_x4"),
        "first14_quarter_discrepancy_min":winners_min("first14_quarter_l1_discrepancy_x4"),
        "first7_cache_lines_max":winners_max("first7_cache_lines_touched"),
        "first14_cache_lines_max":winners_max("first14_cache_lines_touched"),
    },
    "g7":next(r for r in rows if r["g"]==7),
    "claim_boundary":(
        "Structural closure metrics follow from independently derived DATA/INFO byte counts. "
        "Prefix dispersion/cache-line metrics are diagnostics only and are not MPRC theorems. "
        "No composite ranking is asserted."
    )
}

root=Path(__file__).resolve().parents[1]
outdir=root/"results"
outdir.mkdir(exist_ok=True)
json_path=outdir/"generator64_exhaustive_sweep.json"
csv_path=outdir/"generator64_exhaustive_sweep.csv"
json_path.write_text(json.dumps({"summary":summary,"rows":rows},indent=2),encoding="utf-8")

fields=[k for k,v in rows[0].items() if not isinstance(v,list)]
with csv_path.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k:r[k] for k in fields})

print(json.dumps(summary,indent=2))
