"""Coprime MPRC configuration finite gate RC1.

Verifies:
- current discovered configurations 64x157 and 128x113
- power-of-two x coprime rule
- CRT coordinate bijection
- GEN7 full-cycle traversal
- rejects 64x256 as a current valid configuration

This is a foundation gate, not a benchmark.
"""

from __future__ import annotations

from math import gcd
import json
from pathlib import Path

GEN=7

CONFIGS=[
    (64,157),
    (128,113),
]


def is_power_of_two(x:int)->bool:
    return x>0 and (x & (x-1))==0


def crt_map_gate(H:int,W:int):
    assert is_power_of_two(H)
    assert gcd(H,W)==1

    seen=set()
    for t in range(H*W):
        seen.add((t%H,t%W))
    assert len(seen)==H*W

    return {
        "pass":True,
        "H":H,
        "W":W,
        "states":H*W,
        "crt_pairs_unique":len(seen),
    }


def gen7_gate(H:int,W:int):
    assert gcd(GEN,H)==1
    assert gcd(GEN,W)==1
    assert gcd(GEN,H*W)==1

    seen=set()
    for t in range(H*W):
        seen.add(((GEN*t)%H,(GEN*t)%W))
    assert len(seen)==H*W

    return {
        "pass":True,
        "GEN":GEN,
        "H":H,
        "W":W,
        "states":H*W,
        "gen7_pairs_unique":len(seen),
    }


def rejected_64x256_gate():
    H,W=64,256
    assert is_power_of_two(H)
    assert gcd(H,W)==64

    seen={(t%H,t%W) for t in range(H*W)}
    # Pair count collapses to lcm(H,W)=256, not 16384.
    assert len(seen)==256
    assert len(seen)<H*W

    return {
        "pass":True,
        "H":H,
        "W":W,
        "states_nominal":H*W,
        "gcd":gcd(H,W),
        "joint_cycle_states":len(seen),
        "valid_current_mprc_configuration":False,
    }


def budget_gate():
    assert 15*128==1920
    assert 15*256==3840
    assert 3840==2*1920

    return {
        "pass":True,
        "information_half":1920,
        "information_full":3840,
        "full_budget_mapped_to_valid_config":False,
    }


def main():
    report={
        "name":"Coprime MPRC Configurations RC1",
        "rule":{
            "height":"2^n",
            "width":"coprime to height",
            "equivalent_for_power_of_two_height":"width odd",
        },
        "discovered_configurations":[],
        "rejected_arithmetic_envelope":rejected_64x256_gate(),
        "information_budgets":budget_gate(),
        "status_boundary":{
            "64x157":"DISCOVERED / CURRENT",
            "128x113":"DISCOVERED / CURRENT",
            "64x256":"ARITHMETIC ENVELOPE ONLY",
            "15x256_information":"EXACT COUNT / UNMAPPED TO TESTED CONFIG",
        },
        "pass":True,
    }

    for H,W in CONFIGS:
        assert is_power_of_two(H)
        assert gcd(H,W)==1
        row={
            "shape":[H,W],
            "states":H*W,
            "gcd":gcd(H,W),
            "crt":crt_map_gate(H,W),
            "gen7":gen7_gate(H,W),
        }
        report["discovered_configurations"].append(row)

    out=Path("results/coprime_mprc_configurations_rc1.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
