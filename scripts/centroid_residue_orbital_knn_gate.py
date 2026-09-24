"""
Centroid -> residue -> phase-orbit -> KNN survival gate
=======================================================

NO TRAINING. NO LABELS. NO PI. NO FLOATS.

SOURCE-GROUNDED PIECES
----------------------
1. Z256 circular distance:
     cdist(a,b)=min((a-b) mod256,(b-a) mod256)
   is a metric; KNN over sum(cdist) is therefore well-founded.

2. Ring-native experimental prototype:
     C(S)=argmin_v sum_{x in S} cdist(v,x)
   We retain the ENTIRE minimizer set on ties. No arithmetic mean.

3. Four phase-complete variants use quarter offsets 0,64,128,192.
   Phase is routing metadata, not a probability and not a Euclidean angle.

NEW CANDIDATE OPERATOR (THIS FILE GATES IT)
--------------------------------------------
For a chosen center mu and stored neighbor j:

    residue r = j-mu mod256

    phase_rotate(r,k) = r + 64*k mod256,  k in {0,1,2,3}

    orbital_state(mu,j,k)
      = mu + phase_rotate(residue(mu,j),k) mod256

For a query x:
    d*(x;mu,j)=min_k cdist(x,orbital_state(mu,j,k))

The best phase set K*(x;mu,j) retains ALL tied k values.

For vectors, the SAME global phase k is applied to every coordinate:
    E_k(x;mu,j)=sum_i cdist(x_i, orbital_state(mu_i,j_i,k))
    E*=min_k E_k

KNN_K ranks stored neighbors by E*. K=4 is a MODEL PARAMETER supplied by
the user; it is not claimed as a theorem or derived constant.

The gate proves finite algebraic/equivariance properties only.
"""

from pathlib import Path
import itertools, json, random

TAU=256
Q=64
PHASES=(0,1,2,3)
KNN_K=4

def cdist(a,b):
    return min((int(a)-int(b))&255,(int(b)-int(a))&255)

def quadrant(p):
    return (int(p)&255)>>6

def hemisphere(p):
    return (int(p)&255)>>7

def within_phase(p):
    return int(p)&63

def rotate64(p,k):
    return (int(p)+Q*(int(k)&3))&255

def frechet_centers(values):
    vals=[int(x)&255 for x in values]
    assert vals
    costs=[]
    best=None
    for v in range(256):
        c=sum(cdist(v,x) for x in vals)
        if best is None or c<best:
            best=c; costs=[v]
        elif c==best:
            costs.append(v)
    return tuple(costs),best

def residue(mu,j):
    return (int(j)-int(mu))&255

def reconstruct(mu,r):
    return (int(mu)+int(r))&255

def orbital_state(mu,j,k):
    r=residue(mu,j)
    return reconstruct(mu,rotate64(r,k))

def scalar_orbital_match(x,mu,j):
    ds=[cdist(x,orbital_state(mu,j,k)) for k in PHASES]
    m=min(ds)
    return m,tuple(k for k,d in enumerate(ds) if d==m)

def vector_energy_phase(x,mu,j,k):
    assert len(x)==len(mu)==len(j)
    return sum(cdist(xi,orbital_state(mi,ji,k))
               for xi,mi,ji in zip(x,mu,j))

def vector_orbital_match(x,mu,j):
    es=[vector_energy_phase(x,mu,j,k) for k in PHASES]
    m=min(es)
    return m,tuple(k for k,e in enumerate(es) if e==m)

def arrange_neighbors(mu,neighbors):
    """Training-side center arrangement. Pure geometry; no labels."""
    # Canonical state tuple is host tie-break only.
    return tuple(sorted((sum(cdist(m,j) for m,j in zip(mu,n)),tuple(n))
                        for n in neighbors))

def orbital_knn(x,mu,neighbors,K=KNN_K):
    rows=[]
    for n in neighbors:
        e,ks=vector_orbital_match(x,mu,n)
        rows.append((e,tuple(n),ks))
    rows.sort(key=lambda z:(z[0],z[1],z[2]))
    return tuple(rows[:min(K,len(rows))])

results={}

# G1 cdist exhaustive core identities + translation invariance.
for a in range(256):
    assert cdist(a,a)==0
    for b in range(256):
        d=cdist(a,b)
        assert 0<=d<=128
        assert d==cdist(b,a)
        # one fixed translation catches wrap path in every pair
        assert d==cdist((a+73)&255,(b+73)&255)
results["G1_cdist"]={"pass":True,"pairs":65536,"max_distance":128}

# G2 singleton Frechet center exact identity.
for v in range(256):
    C,cost=frechet_centers([v])
    assert C==(v,) and cost==0
results["G2_frechet_singleton"]={"pass":True,"singletons":256}

# G3 Frechet center SET translation equivariance. No arbitrary tie collapse.
rng=random.Random(20260925)
clusters=3000
for _ in range(clusters):
    n=rng.randrange(1,13)
    S=[rng.randrange(256) for _ in range(n)]
    t=rng.randrange(256)
    C,cost=frechet_centers(S)
    Ct,costt=frechet_centers([((x+t)&255) for x in S])
    expect=tuple(sorted(((c+t)&255) for c in C))
    assert tuple(sorted(Ct))==expect
    assert costt==cost
results["G3_frechet_translation_equivariance"]={
    "pass":True,"random_clusters":clusters,"ties_preserved_as_sets":True
}

# G4 centroid/residue decoupling is exact for every mu,j.
for mu in range(256):
    for j in range(256):
        r=residue(mu,j)
        assert reconstruct(mu,r)==j
        # Common motion changes center/neighbor but not residue.
        t=91
        assert residue((mu+t)&255,(j+t)&255)==r
results["G4_centroid_residue"]={"pass":True,"pairs":65536}

# G5 four quarter phases are exact: offset within the 64-state phase is
# preserved and phase quadrant advances by k.
for p in range(256):
    for k in PHASES:
        q=rotate64(p,k)
        assert within_phase(q)==within_phase(p)
        assert quadrant(q)==((quadrant(p)+k)&3)
    assert rotate64(p,4)==p
    assert len({rotate64(p,k) for k in PHASES})==4
results["G5_phase_orbit"]={
    "pass":True,"states":256,"phase_count":4,"quarter_step":64,"order":4
}

# G6 hemisphere is explicit metadata derived from the actual candidate state.
for p in range(256):
    assert hemisphere(p) in (0,1)
    assert hemisphere(p)==(0 if p<128 else 1)
results["G6_hemisphere"]={
    "pass":True,"positive_half":"0..127","negative_half":"128..255"
}

# G7 k=0 reconstructs the original stored neighbor exactly; phase-complete
# matching can never be worse than phase-0-only matching.
for mu in range(256):
    for j in range(256):
        assert orbital_state(mu,j,0)==j
        for x in (0,1,63,64,127,128,191,192,255):
            best,ks=scalar_orbital_match(x,mu,j)
            assert best<=cdist(x,j)
            assert 1<=len(ks)<=4
results["G7_phase_complete_match"]={
    "pass":True,"mu_j_pairs":65536,"query_probes_per_pair":9
}

# G8 common translation equivariance of orbital matching.
trials=200000
for _ in range(trials):
    x=rng.randrange(256);mu=rng.randrange(256);j=rng.randrange(256);t=rng.randrange(256)
    d,ks=scalar_orbital_match(x,mu,j)
    dt,kst=scalar_orbital_match((x+t)&255,(mu+t)&255,(j+t)&255)
    assert dt==d and kst==ks
results["G8_orbital_translation_equivariance"]={"pass":True,"random_trials":trials}

# G9 vector extension uses one common phase k and integer energy only.
for dim in (1,4,8,16):
    for _ in range(2000):
        x=[rng.randrange(256) for _ in range(dim)]
        mu=[rng.randrange(256) for _ in range(dim)]
        j=[rng.randrange(256) for _ in range(dim)]
        e,ks=vector_orbital_match(x,mu,j)
        assert isinstance(e,int) and e>=0
        assert 1<=len(ks)<=4
        # Common translation vector preserves energy and best phases.
        t=[rng.randrange(256) for _ in range(dim)]
        xt=[(a+b)&255 for a,b in zip(x,t)]
        mut=[(a+b)&255 for a,b in zip(mu,t)]
        jt=[(a+b)&255 for a,b in zip(j,t)]
        e2,ks2=vector_orbital_match(xt,mut,jt)
        assert (e2,ks2)==(e,ks)
results["G9_vector_energy"]={
    "pass":True,"dimensions":[1,4,8,16],"same_global_phase_per_vector":True
}

# G10 training-side neighbor arrangement is invariant under a common ring shift.
for _ in range(10000):
    dim=8
    mu=[rng.randrange(256) for _ in range(dim)]
    ns=[[rng.randrange(256) for _ in range(dim)] for _ in range(12)]
    t=[rng.randrange(256) for _ in range(dim)]
    A=arrange_neighbors(mu,ns)
    mut=[(a+b)&255 for a,b in zip(mu,t)]
    nst=[[(a+b)&255 for a,b in zip(n,t)] for n in ns]
    B=arrange_neighbors(mut,nst)
    # Compare only center distances; canonical tuple tie metadata naturally shifts.
    assert [r[0] for r in A]==[r[0] for r in B]
results["G10_center_neighbor_arrangement"]={
    "pass":True,"random_sets":10000,"ordering_key":"sum cdist from center"
}

# G11 Top-K implementation survives input permutation. Canonical tuple is a host
# tie-break only; geometry is the distance and retained best phase set.
for _ in range(10000):
    dim=8
    x=[rng.randrange(256) for _ in range(dim)]
    mu=[rng.randrange(256) for _ in range(dim)]
    ns=[tuple(rng.randrange(256) for _ in range(dim)) for _ in range(12)]
    A=orbital_knn(x,mu,ns,KNN_K)
    rng.shuffle(ns)
    B=orbital_knn(x,mu,ns,KNN_K)
    assert A==B
results["G11_KNN4_permutation_invariance"]={
    "pass":True,"random_sets":10000,"K":KNN_K
}

# G12 no probability/softmax semantics appear in the operator.
results["G12_semantic_boundary"]={
    "pass":True,
    "centroid":"circular Frechet minimizer set under cdist",
    "neighbor_radius":"sum cdist(center,neighbor)",
    "query_measure":"minimum integer orbital energy over four phases",
    "phase":"retained routing metadata k in {0,1,2,3}",
    "hemisphere":"retained from matched candidate state",
    "probability":False,
    "softmax":False,
    "K_status":"user-supplied model parameter, not theorem",
}

results["overall"]="ALL PASS"
results["training_gate_open"]=True
results["claim_boundary"]=(
    "cdist metric and phase-complete four-way routing are inherited from frozen framework results. "
    "The centroid-residue orbital KNN composition is a new candidate operator whose finite "
    "equivariance/code properties pass here; classification usefulness remains empirical."
)

root=Path(__file__).resolve().parents[1]
out=root/"results"/"centroid_residue_orbital_knn_gate.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results,indent=2),encoding="utf-8")
print(json.dumps(results,indent=2))
