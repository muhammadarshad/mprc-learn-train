"""v19 — Resolved Solution Memory + Constructive Constraint Solver.

This is NOT a memorisation-vs-generalisation benchmark.

MPRC interpretation
===================
Memory contains RESOLVED states. Reusing one is desirable.

For the constraint

    a1 + a2 = T

define signed reaction residual

    r = a1 + a2 - T

and MEASURE

    E = |r|

Then:

OBSERVE/IDENTIFY
    read target T and any current exclusions/constraints

BIND
    keep the ordered feature pair (a1,a2)

REACT
    compute signed residual r

MEASURE
    E=|r|

MOVE
    if E>0: move NORMAL to the solution manifold
    if E=0 but candidate is excluded: move TANGENT along the exact-solution manifold

SELECT
    choose exact E=0 if available; otherwise minimum E

LUT
    store a newly resolved exact state and its resolution route

Integer representation
======================
Use deci-units only. No float arithmetic.

Examples:
    2.0 -> 20
    1.5 -> 15
    2.5 -> 25
    4.0 -> 40

Therefore:
    15+25-40 = 0     exact
    14+25-40 = -1    near-exact, distance 0.1

The finite benchmark checks:
1. remembered exact selection
2. exclusion forces constructive tangent MOVE to another exact solution
3. unseen target is constructed from scratch
4. near-exact state is repaired by residual-directed normal MOVE
5. memory-only control fails when its stored answer is excluded
6. constructive solver succeeds under the same constraint
7. exhaustive finite survival over many targets/exclusion depths

The solver is intentionally small and exact so the operation semantics are visible.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

SCALE=10
MAX_VALUE=80
UNIT=SCALE  # preferred human-scale tangent move = 1.0


def fmt(x:int)->str:
    sign="-" if x<0 else ""
    u=abs(int(x))
    return f"{sign}{u//SCALE}.{u%SCALE}"


def react(a:int,b:int,target:int)->int:
    return int(a)+int(b)-int(target)


def measure(a:int,b:int,target:int)->int:
    return abs(react(a,b,target))


@dataclass(frozen=True)
class Resolution:
    target:int
    a1:int
    a2:int
    energy:int
    source:str
    route:tuple[tuple[int,int,int], ...]  # (a1,a2,signed residual)

    def exact(self)->bool:
        return self.energy==0


class ResolvedLUT:
    def __init__(self):
        self.by_target:dict[int,list[Resolution]]={}

    def remember(self,res:Resolution)->None:
        if not res.exact():
            return
        bucket=self.by_target.setdefault(res.target,[])
        if not any((r.a1,r.a2)==(res.a1,res.a2) for r in bucket):
            bucket.append(res)

    def exact_candidates(self,target:int,excluded:set[tuple[int,int]]):
        return [
            r for r in self.by_target.get(int(target),[])
            if (r.a1,r.a2) not in excluded and r.energy==0
        ]

    def any_exact(self,target:int):
        return list(self.by_target.get(int(target),[]))


class ConstructiveSolver:
    def __init__(self,max_value:int=MAX_VALUE):
        self.max_value=int(max_value)
        self.lut=ResolvedLUT()

    def seed(self,target:int,a1:int,a2:int):
        e=measure(a1,a2,target)
        assert e==0
        self.lut.remember(
            Resolution(
                target,a1,a2,e,"seed",
                ((a1,a2,react(a1,a2,target)),)
            )
        )

    def _valid(self,a,b):
        return 0<=a<=self.max_value and 0<=b<=self.max_value

    def _balanced_start(self,target:int):
        # Start from the origin and let normal MOVE construct the state.
        return 0,0

    def _normal_move(self,a:int,b:int,target:int,step_index:int):
        """One unit move toward residual zero.

        Alternate axes so construction does not collapse every target to (T,0).
        Prefer the axis with smaller coordinate when both can move.
        """
        r=react(a,b,target)
        if r==0:
            return a,b

        if r<0:
            # Need to increase total by one deci-unit.
            choices=[]
            if a<self.max_value:
                choices.append((a+1,b))
            if b<self.max_value:
                choices.append((a,b+1))
        else:
            # Need to decrease total by one deci-unit.
            choices=[]
            if a>0:
                choices.append((a-1,b))
            if b>0:
                choices.append((a,b-1))

        assert choices

        # Prefer balance, then alternate orientation deterministically.
        choices.sort(key=lambda z:(abs(z[0]-z[1]), (z[0] if step_index%2==0 else z[1])))
        return choices[0]

    def _tangent_candidates(self,a:int,b:int):
        """Exact-manifold motion preserves a+b.

        Prefer whole-unit alternatives first (e.g. 2+2 -> 3+1),
        then deci-unit moves if whole-unit motion is blocked.
        """
        out=[]
        for q in (UNIT,1):
            for da,db in ((q,-q),(-q,q)):
                x,y=a+da,b+db
                if self._valid(x,y):
                    out.append((x,y))
        # unique, stable
        seen=set();ans=[]
        for z in out:
            if z not in seen:
                seen.add(z);ans.append(z)
        return ans

    def solve(self,target:int,excluded:set[tuple[int,int]]|None=None)->Resolution:
        target=int(target)
        excluded=set() if excluded is None else set(excluded)

        # SELECT from resolved experience first.
        remembered=self.lut.exact_candidates(target,excluded)
        if remembered:
            # Balanced/short response is default resolved experience.
            remembered=sorted(remembered,key=lambda r:(abs(r.a1-r.a2),r.a1,r.a2))
            r=remembered[0]
            return Resolution(
                r.target,r.a1,r.a2,0,"memory-select",
                ((r.a1,r.a2,0),)
            )

        # If target has an exact remembered state but all are excluded,
        # MOVE tangent on the exact solution manifold.
        prior=self.lut.any_exact(target)
        if prior:
            start=sorted(prior,key=lambda r:(abs(r.a1-r.a2),r.a1,r.a2))[0]
            a,b=start.a1,start.a2
            route=[(a,b,0)]
            visited={(a,b)}

            frontier=[(a,b)]
            while frontier:
                x,y=frontier.pop(0)
                for nx,ny in self._tangent_candidates(x,y):
                    if (nx,ny) in visited:
                        continue
                    visited.add((nx,ny))
                    assert react(nx,ny,target)==0
                    route.append((nx,ny,0))

                    if (nx,ny) not in excluded:
                        res=Resolution(
                            target,nx,ny,0,"construct-tangent",tuple(route)
                        )
                        self.lut.remember(res)
                        return res

                    frontier.append((nx,ny))

        # No resolved state exists: construct by normal MOVE from origin.
        a,b=self._balanced_start(target)
        route=[(a,b,react(a,b,target))]
        steps=0

        while measure(a,b,target)>0:
            a,b=self._normal_move(a,b,target,steps)
            steps+=1
            route.append((a,b,react(a,b,target)))
            assert steps<=2*self.max_value+2

        # Exact but possibly excluded. If excluded, seed it temporarily then
        # continue tangent construction.
        res=Resolution(target,a,b,0,"construct-normal",tuple(route))

        if (a,b) not in excluded:
            self.lut.remember(res)
            return res

        self.lut.remember(res)
        return self.solve(target,excluded)

    def repair(self,a:int,b:int,target:int)->Resolution:
        """Repair a supplied near-exact observation using residual-directed MOVE."""
        a=int(a);b=int(b);target=int(target)
        route=[(a,b,react(a,b,target))]
        steps=0

        while measure(a,b,target)>0:
            a,b=self._normal_move(a,b,target,steps)
            steps+=1
            route.append((a,b,react(a,b,target)))
            assert steps<=2*self.max_value+2

        res=Resolution(target,a,b,0,"repair-near-exact",tuple(route))
        self.lut.remember(res)
        return res


def memory_only(lut:ResolvedLUT,target:int,excluded:set[tuple[int,int]]):
    c=lut.exact_candidates(target,excluded)
    if not c:
        return None
    return sorted(c,key=lambda r:(abs(r.a1-r.a2),r.a1,r.a2))[0]


def run_demo():
    s=ConstructiveSolver()
    s.seed(40,20,20)  # 2+2=4

    first=s.solve(40)
    assert (first.a1,first.a2)==(20,20)
    assert first.source=="memory-select"

    # "No, give me another solution."
    second=s.solve(40,{(20,20)})
    assert second.energy==0
    assert (second.a1,second.a2)!=(20,20)

    # Preferred whole-unit tangent should make 3+1 or 1+3 available immediately.
    assert (second.a1,second.a2) in {(30,10),(10,30)}

    # Near-exact explicit example from the owner.
    near_before={
        "a1":14,
        "a2":25,
        "target":40,
        "signed_residual":react(14,25,40),
        "energy":measure(14,25,40),
    }
    assert near_before["signed_residual"]==-1
    assert near_before["energy"]==1

    repaired=s.repair(14,25,40)
    assert repaired.energy==0
    assert repaired.a1+repaired.a2==40

    # Unseen target: no LUT entry before solving.
    assert 37 not in s.lut.by_target
    unseen=s.solve(37)
    assert unseen.energy==0
    assert unseen.a1+unseen.a2==37
    assert unseen.source=="construct-normal"

    return {
        "first_resolved":{
            "equation":f"{fmt(first.a1)} + {fmt(first.a2)} = {fmt(first.target)}",
            "source":first.source,
        },
        "excluded_first_constructed_second":{
            "equation":f"{fmt(second.a1)} + {fmt(second.a2)} = {fmt(second.target)}",
            "source":second.source,
            "route_length":len(second.route),
        },
        "near_exact_before":{
            "equation":f"{fmt(14)} + {fmt(25)} = {fmt(39)}",
            "target":fmt(40),
            "signed_residual_deci":-1,
            "energy":"0.1",
        },
        "near_exact_repaired":{
            "equation":f"{fmt(repaired.a1)} + {fmt(repaired.a2)} = {fmt(repaired.target)}",
            "route_length":len(repaired.route),
        },
        "unseen_target_constructed":{
            "target":fmt(37),
            "equation":f"{fmt(unseen.a1)} + {fmt(unseen.a2)} = {fmt(unseen.target)}",
            "route_length":len(unseen.route),
        },
    }


def exhaustive_gate():
    """Finite survival over targets and excluded exact solutions."""
    memory_control_fail=0
    constructive_success=0
    unseen_success=0
    repair_success=0
    exact_checks=0

    # Seed only even targets with their balanced exact resolution.
    solver=ConstructiveSolver()
    for T in range(0,MAX_VALUE+1,2):
        a=T//2
        b=T-a
        solver.seed(T,a,b)

    # A) unseen odd targets must be constructed exactly.
    for T in range(1,MAX_VALUE+1,2):
        assert T not in solver.lut.by_target
        r=solver.solve(T)
        assert r.energy==0 and r.a1+r.a2==T
        unseen_success+=1
        exact_checks+=1

    # B) For every target, exclude currently remembered default.
    # Memory-only should fail if that target has only one stored state;
    # constructive solver must find another whenever the finite domain permits.
    for T in range(1,MAX_VALUE):
        bucket=solver.lut.any_exact(T)
        assert bucket

        default=sorted(bucket,key=lambda r:(abs(r.a1-r.a2),r.a1,r.a2))[0]
        ex={(default.a1,default.a2)}

        if memory_only(solver.lut,T,ex) is None:
            memory_control_fail+=1

        r=solver.solve(T,ex)
        assert r.energy==0
        assert r.a1+r.a2==T
        assert (r.a1,r.a2) not in ex
        constructive_success+=1
        exact_checks+=1

    # C) Near-exact repair for every interior target and perturbation -5..+5.
    for T in range(5,MAX_VALUE-4):
        a=T//2
        b=T-a

        for delta in range(-5,6):
            if delta==0:
                continue
            aa=a+delta
            if not 0<=aa<=MAX_VALUE:
                continue

            e0=measure(aa,b,T)
            if e0==0:
                continue

            r=solver.repair(aa,b,T)
            assert r.energy==0
            assert r.a1+r.a2==T
            repair_success+=1
            exact_checks+=1

    return {
        "pass":True,
        "targets_domain":[0,MAX_VALUE],
        "unseen_target_constructive_success":unseen_success,
        "excluded_default_constructive_success":constructive_success,
        "memory_only_failures_when_only_answer_excluded":memory_control_fail,
        "near_exact_repairs":repair_success,
        "exact_checks":exact_checks,
        "lut_targets_resolved":len(solver.lut.by_target),
        "lut_exact_states_total":sum(len(v) for v in solver.lut.by_target.values()),
    }


def approximation_gate():
    """Demonstrate SELECT when exact solution is forbidden by the observation lattice.

    Restrict both factors to multiples of 0.2 => even deci-units.
    Target 3.9 => 39 is odd, so exact equality is impossible.
    MEASURE must select energy 1 => 0.1.
    """
    target=39
    vals=range(0,MAX_VALUE+1,2)
    rows=[]

    for a in vals:
        for b in vals:
            E=measure(a,b,target)
            rows.append((E,abs(a-b),a,b))

    rows.sort()
    E,_,a,b=rows[0]

    assert E==1
    assert (a+b)%2==0
    assert a+b in (38,40)

    return {
        "pass":True,
        "target":fmt(target),
        "factor_quantum":"0.2",
        "exact_solution_possible":False,
        "selected":[fmt(a),fmt(b)],
        "selected_sum":fmt(a+b),
        "energy":"0.1",
    }


demo=run_demo()
exhaustive=exhaustive_gate()
approx=approximation_gate()

report={
    "model":"v19-resolved-solution-memory-and-construction",
    "representation":{
        "scale":"integer deci-units; no floats",
        "constraint":"F(a1,a2;T)=a1+a2-T",
        "BIND":"ordered pair (a1,a2)",
        "REACT":"signed residual F",
        "MEASURE":"absolute residual |F|",
        "MOVE_normal":"change residual toward zero",
        "MOVE_tangent":"move along exact manifold a1+a2=T when a remembered exact state is excluded",
        "SELECT":"resolved exact state first; otherwise construct; if exact impossible choose minimum MEASURE",
        "LUT":"stores resolved exact states and their resolution route",
    },
    "demo":demo,
    "exhaustive_gate":exhaustive,
    "approximation_gate":approx,
    "interpretation":{
        "memory":"resolved experience; desirable fast path",
        "construction":"invoked when current observation rejects/does not contain a usable resolved state",
        "generalisation":"not treated as opposite of memory; new valid resolution is constructed and verified by MEASURE",
    },
    "claim_boundary":(
        "v19 is an exact finite constraint-solving mechanics gate. It does not yet "
        "encode arbitrary language/problem semantics, nor does it prove that all MPRC "
        "reasoning reduces to a1+a2=T. It validates the intended memory -> constraint -> "
        "construct -> measure -> select -> remember cycle."
    ),
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"resolved_solution_constructive_v19.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
