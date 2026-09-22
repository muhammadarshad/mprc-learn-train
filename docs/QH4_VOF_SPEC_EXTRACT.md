# QH4 VOF interface reconstruction

> Text extract of the author-supplied source document used for this research repo.

<PARSED TEXT FOR PAGE: 1 / 10>
Volume-of-Fluid Interface Reconstruction on the QH4 Ring
Degenerate Orientations as Vacuum Nodes, and a Depth Law for Quantised Accuracy
Muhammad Arshad Independent Researcher
Abstract—Piecewise-linear interface calculation (PLIC) in three-dimensional volume-of-fluid 
(VOF) methods carries a structural weakness: the volume function has denominator , which 
vanishes identically at axis-aligned orientations. Production codes detect this with a runtime 
tolerance and branch to a special case. We present an interface representation on the QH4 ring 
— a 256-state cyclic structure with four distinguished states at , previously introduced as 
vacuum nodes — in which those four states are precisely the degenerate orientations, and in 
which the transition operator provably cannot reach them. The operator advances the ring 
coordinate by two; over this is exactly multiplication by , and we prove it is the minimal step 
whose orbit is disjoint from the degenerate set, that orbit being the complete class of quadratic 
non-residues traversed as a single cycle. Working over rather than makes the accumulation 
encoding of the interface 4-vector bijective at , where the same encoding over the ring loses two
bits. Under the standard normalisation the accumulation equals , isolating the plane offset 
exactly. The action leaves invariant to the last bit, which is the 48-fold symmetry that reduces 
the offset-inversion table to 25 KB. In compiled tests the scheme conserves volume exactly and 
returns bit-identical fields across every domain decomposition, where the floating-point 
reference returns seven distinct fields out of eight with a cellwise spread one third the size of 
the discretisation error itself. On the Rider–Kothe reversed vortex we then derive and confirm a
ring-depth law: quantisation is accuracy-neutral against float64 exactly when the number of 
ring levels satisfies for a scheme of order — one additional bit per doubling of grid resolution 
at the measured . The law is fitted on and confirmed out of sample at . A previously reported 
sub-quantum census is retracted.
Index Terms—Volume of fluid, PLIC, interface reconstruction, exact integer arithmetic, finite 
field, reproducibility, computational fluid dynamics.
I. Introduction
A. The degeneracy
A PLIC reconstruction represents the interface in a cell by a plane , with the volume fraction
where and ( normalisation). The denominator vanishes whenever any component of is zero 
— that is, at every axis-aligned interface. The cubic collapses to a lower-order branch and 
implementations special-case it, typically on a floating-point tolerance test.
Verified: falls as .
<PARSED TEXT FOR PAGE: 2 / 10>
B. The contribution
The QH4 ring [Arshad, 2025–2026] carries 256 states in four quadrants of 64, with four 
distinguished states at termed vacuum nodes. This paper establishes three things.
1. The vacuum nodes are the degenerate orientations. Under the ring’s phase 
assignment they are exactly the axis-aligned normals, where (1) is singular (Section II).
2. The transition operator cannot reach them. Advancing the ring coordinate by two is
multiplication by over , and its orbit is provably disjoint from the node set (Section III).
3. Consequences for the VOF step, quantified. Exact interface encoding, an offset 
inversion of eight integer comparisons, and volume conservation that is bit-identical 
across every domain decomposition where float64 gives a different field for each 
(Sections IV–VI). The accuracy cost of working at finite ring depth is then derived 
rather than merely measured: neutrality against float64 requires for a scheme of order ,
which at the measured is one extra bit of depth per doubling of resolution. Fitted on 
three grids, the law is confirmed on a fourth it never saw (Section VII).
The degeneracy is not detected and handled; it is made unreachable.
II. Vacuum Nodes Are the Degenerate Orientations
Definition 1 (QH4 ring). 256 states indexed by , four quadrants of 64, vacuum nodes at .
Proposition 1. Under orientation, the vacuum nodes are the axis-aligned normals:
orientation
0
64
128
192
Verified. At each, some component of is zero, so and (1) is undefined. The vacuum node is the
PLIC degeneracy; the two were named independently and are the same object.
Proposition 2 (field form). Over with primitive root and the discrete logarithm, the nodes 
are the -torsion subgroup: , , , , of orders . (Verified.) In particular is an exact integer 
congruence, requiring no phase map.
III. The Transition Operator
Definition 2. .
Proposition 3. In , is multiplication by , where and is a primitive root modulo 257. (Verified: , ,
with the node traversed but never occupied.)
<PARSED TEXT FOR PAGE: 3 / 10>
Theorem 1. is the minimal exponent step whose orbit is disjoint from the vacuum-node set, 
and its orbit is the complete set of 128 odd exponents, traversed as one cycle.
Proof. The nodes are all even. From an odd exponent, preserves parity, so no iterate is even and
none is a node. The orbit from has length 128, is entirely odd, and covers every odd residue. 
Exhaustive check of smaller steps: reaches a node after 63 states.
Corollary 1. The odd exponents are the quadratic non-residues of ; the even ones, including all 
four nodes, are the squares. Since is a square, multiplication by preserves the residue class. 
(Verified.)
Corollary 2 (closure). , , , . Each steps over one node; the cycle closes on the identity, not on 0, 
since 0 is itself a node.
Remark. In VOF terms: an interface orientation transported by never lands on an orientation 
at which (1) is singular. No tolerance test is required because the singular configuration is not 
in the reachable set.
IV. Interface Encoding
The cell state is the 4-vector . The accumulation encoding is
Proposition 4. Over , (2) is bijective for every , since every such is invertible. (Verified at : 
200,000 random 4-vectors, zero failures.)
This is why the field matters. Over the same map has ; at , and it is four-to-one, losing two 
bits — verified by exhaustive enumeration. VOF forces .
Proposition 5. Under normalisation, : the accumulation is the plane offset, with orientation 
carried by the keys. (Verified: 100,000 states, zero failures.)
V. The Volume Function by Integer Difference Table
Proposition 6. . For the third difference is 6 — and (1) is cubic in , so on a branch is exactly 
constant and exactly zero. (Verified in exact rational arithmetic: for , .)
Hence the branch is generated by three integer accumulators,
reproducing the closed form exactly over 9 consecutive steps (verified). Since is monotone in , 
inverting it on the 256-level grid takes exactly 8 integer comparisons, against the cube roots 
and inverse trigonometric functions of the analytic route.
<PARSED TEXT FOR PAGE: 4 / 10>
Proposition 7 ( invariance). is invariant under conjugation by any element: over 20,000 
trials — exactly zero, since conjugation permutes and sign-flips the normal components and 
depends only on sorted .
This invariance is the 48-fold symmetry that reduces the offset-inversion table: 5-bit normals 
give 102 directions 256 levels 25 KB, L1-resident. The table reduction is a consequence of the 
action, not a separate optimisation.
VI. Conservation and Reproducibility
Two independent tests, both on schemes whose correctness is established in Section VII.
3-D unsplit advection. Compiled C, , 40 steps, CFL (non-dyadic). Five domain decompositions;
identical scheme and data throughout, only the arithmetic and the block partition vary.
arithmetic
distinct final fields (5 
decompositions) volume drift
float64, naive 5
float64, Kahan￾compensated
5
float64, fixed-point 
accumulation
1 0
QH4 ring 1 0 (exact)
2-D Rider–Kothe vortex. , 512 steps, eight decompositions (block sizes ), with the standard 
global mass-fixing step after each sweep pair — the step that forces a whole-grid reduction and 
therefore makes the result depend on the partition.
arithmetic
distinct final fields (8 
decompositions) max cellwise spread
float64 7
QH4 ring 1 0 levels (bit-identical)
The float spread is — four orders of magnitude above machine epsilon, and one third of the 
discretisation error itself at this resolution (, Section VII). Decomposition non-determinism in a
long VOF run is not a rounding curiosity; it is a term of the same order as the physics being 
resolved.
Two findings from the 3-D table. Kahan compensation does not give decomposition 
invariance — five distinct fields, as many as naive — because the compensation term is itself 
order-dependent; compensated and reproducible summation are routinely conflated and are 
not the same. Fixed-point accumulation does, and the mechanism is instructive: the 
established route to reproducible floating-point summation is to accumulate in integers.
<PARSED TEXT FOR PAGE: 5 / 10>
The ring obtains both properties structurally, at every operation, without conversion or a 
separate accumulator. That is a claim about where the property lives, not about who can reach 
it.
VII. Accuracy, and the Ring-Depth Law
Section VI shows the ring is reproducible. Reproducibly wrong is worthless, so the decisive 
question is what the quantisation costs. This section answers it, and the answer is a scaling law 
with a free parameter that was fitted on three resolutions and then tested on a fourth.
A. Benchmark and reference implementation
The Rider–Kothe reversed single vortex is the standard VOF accuracy test. The velocity field
stretches a circular blob into a thin filament and reverses at , so the exact field at is the initial 
field and every deviation is error. We use , a disk of radius centred at , CFL , Strang-alternated 
directional splitting with the Weymouth–Scardovelli–Zaleski divergence correction, Youngs 
normals, and .
The float64 reference must be sound before any comparison means anything:
 (float64) order volume drift
32 —
64 1.83
128 1.53
256 1.14 —
Both the error magnitudes and the observed order are in the published range for Youngs￾normal PLIC on this benchmark. The comparison below is therefore against a competent 
reference, not a straw man.
B. Ring depth versus resolution
Let be the number of ring levels ( for the 8-bit QH4 ring). The table gives ; means the 
quantisation costs nothing.
64 1.069 1.902 4.622
128 1.064 1.469 1.918
256 1.015 1.072 1.385
512 0.989 1.050 1.038
1024 1.009 1.002 0.959
2048 0.996 0.996 1.014
<PARSED TEXT FOR PAGE: 6 / 10>
4096 1.003 0.998 1.000
Read down a column: quantisation is invisible until falls below a resolution-dependent 
threshold, then dominates rapidly. Read across a row: the threshold moves right as grows. 
Taking accuracy-neutrality to mean ratio , the boldface entries give the smallest sufficient 
depth, and
Fig. 1. Left: absolute error against the exact reversed-vortex solution; the ring tracks the float64 
reference exactly until its depth runs out, then flattens. Right: the same data as a ratio. 
Neutrality (below the dashed line) holds precisely where . The circled point at is the out-of￾sample confirmation of Section VII-D.
C. Why the law has this form
The result is not a fit looking for a story. Quantising to levels injects a per-cell error uniform 
on , contributing at most to . Only interface cells are affected, of which there are for interface 
length . The run takes steps and the injections are mutually incoherent, so they accumulate as :
The truncation error of a scheme of order is . Neutrality therefore requires
With the measured this is linear in , which is what the table shows, and (7) fixes the constant 
at 8. Equivalently, in bits:
One additional bit of ring depth per doubling of grid resolution. The 8-bit QH4 ring is 
accuracy-neutral to in two dimensions; needs 10 bits, needs 13.
<PARSED TEXT FOR PAGE: 7 / 10>
D. Out-of-sample test
Equations (7)–(10) were fitted on . They predict that at — a resolution not used in the fit — 
depth is neutral and is marginal. Run:
 ratio predicted
256 2.028 fail
512 1.361 fail
1024 1.071 marginal ()
2048 0.993 neutral ()
The prediction holds. This is the strongest form of the claim available: the quantisation cost of 
the ring is not merely measured, it is predictable in advance from the scheme’s order of 
accuracy, and the prediction survives a resolution it was not fitted to.
E. What this settles
The ring is not more accurate than float64 and this paper does not claim it is. It is accuracy￾neutral at a stated, derivable depth, and at that depth it also delivers exact conservation and 
bit-identical results across every domain decomposition (Section VI), neither of which float64 
provides at any precision. The trade is a known number of bits against a property that floating 
point cannot supply.
VIII. Costs and Limitations
1. Depth is a function of resolution, not a constant. Equation (10) is a requirement, not
an option. An 8-bit ring at costs a factor of two in (Section VII-D) and must not be used 
there. Any deployment must state and together.
2. The 2-D law is verified; the 3-D law is not. Equation (9) depends on interface 
measure scaling, which changes from to in three dimensions, predicting — steeper, 
and untested. This is the first thing a referee should ask for and the first thing we would
run next.
3. Retraction. An earlier version of this work reported that 69.3% of partial cells hold 
volume fractions below . That census was taken from an advection routine subsequently
found to lose 40% of the disk’s volume under pure translation; the figure is withdrawn 
and no conclusion rests on it. The accuracy question it was meant to bear on is now 
answered directly in Section VII.
4. Normal estimation is not addressed by the ring. Youngs normals are computed in 
floating point in the reference implementation here. Recovering from neighbouring 
fractions is the dominant error source in most VOF solvers; a fully ring-native normal 
estimator is not part of this paper, and the accuracy results above therefore isolate the 
cost of quantising and alone.
5. Normal-quantisation accuracy is untested, and the 25 KB table-size argument in 
Section V depends on it.
<PARSED TEXT FOR PAGE: 8 / 10>
6. No performance claim. A fair speed comparison requires the Scardovelli–Zaleski 
analytic inversion as the reference, not bisection, and it has not been run. No timing 
figure is quoted anywhere in this paper.
7. Overflow bounds are not derived. Exact modular conservation is physical 
conservation only while nothing wraps.
IX. Relation to Prior Work
PLIC and its analytic offset inversion are due to Scardovelli and Zaleski. Arithmetic over 
appears independently elsewhere — the Fermat Number Transform (Agarwal & Burrus, 1974) 
and the IDEA cipher (Lai & Massey, 1991) — reached for their own reasons; nothing of those 
results is claimed here, and a shared modulus is coexistence, not derivation. The group theory 
invoked is classical.
The QH4 ring, its vacuum nodes, and the state vector are prior work of the author (2025–2026).
What this paper contributes is the identification of the vacuum nodes with the PLIC 
degeneracy, Theorem 1, and the consequences in Sections IV–VII, including the ring-depth law 
(7)–(10) and its out-of-sample confirmation.
Appendix A. Reproducibility and Provenance of Every Numeric Claim
No number in this paper is quoted from a prior draft, a secondary source, or an assistant’s 
summary. Every value below was produced by the named program, run by the author, on the 
machine that produced this manuscript. Programs are supplied with the submission. A claim 
that cannot be regenerated by running the listed command is a claim this paper does not make.
§ Claim Source program Regenerates as
I-A : , , ring_plic.py printed sequence
II anchor axis-aligned 
normal, 4 cases
ring_plic.py §4 anchor=True/False
table
III bridge ; orbit disjoint
from
proof (§III), 
enumerated in 
mprc_core.py
exhaustive over 256 
states
IV ADI encode/decode, 
200,000 trials, 0 
failures
ring_plic.py §1 0 failures
IV , 100,000 trials, 0 
failures
ring_plic.py §2 0 failures
V by 3 integer adds, 
exact 9 steps, ,
ring_plic.py §3 EXACT for 9 steps
V offset inversion, 8 
integer compares, 
mean err
ring_plic.py §5 2000 cases
<PARSED TEXT FOR PAGE: 9 / 10>
§ Claim Source program Regenerates as
V curvature from 
integer 2nd 
differences, 0.00% at
ring_plic.py §6 5-row table
V invariance of , 
20,000 trials, max 
error exactly 0
verify_all.py max = 0
VI 3-D decomposition 
census (5 arithmetics
5 partitions)
vof3d.c 4-row table
VI 2-D decomposition 
census, 7 vs 1 distinct 
fields, spread
repro_vof.c RESULT: line
VII-A float64 and order at accuracy_vof.c ./accuracy_vof 2.0
VII-B ring/float ratio, accuracy_vof.c ./accuracy_vof 2.0 
<S> <N>
VII-D out-of-sample 
confirmation at
accuracy_vof.c ./accuracy_vof 2.0 
2048 256 0.993
Two negative results are recorded here deliberately, because both were reached by 
falsifying claims this work previously advanced.
First, the sub-quantum census of §VIII-3 was withdrawn after the advection routine that 
produced it was found to lose 40% of the test volume under pure translation — a defect 
invisible in the reported statistic and detected only by a translation invariance check that the 
original benchmark did not include. The check is now standard in accuracy_vof.c.
Second, an earlier companion result asserted a dyadic obstruction preventing exact power-of￾two weights in an isotropic velocity set. It is false: verify_all.py locates a counterexample at 
with dyadic weights on shells . The claim is retracted, and the correct statement — that the 
minimal exactly-representable ring is , requiring 12 bits — is what the program actually 
establishes.
Both errors were found by running code against the claim rather than by rereading the 
argument. That is the standard this appendix exists to hold the rest of the paper to.
References
[1] R. Scardovelli and S. Zaleski, “Analytical relations connecting linear interfaces and volume 
fractions in rectangular grids,” J. Comput. Phys., vol. 164, no. 1, pp. 228–237, 2000.
[2] D. L. Youngs, “Time-dependent multi-material flow with large fluid distortion,” in 
Numerical Methods for Fluid Dynamics, Academic Press, 1982.
<PARSED TEXT FOR PAGE: 10 / 10>
[3] M. Owkes and O. Desjardins, “A computational framework for conservative, three￾dimensional, unsplit, geometric transport with application to the volume of fluid (VOF) 
method,” J. Comput. Phys., vol. 270, pp. 587–612, 2014.
[4] R. C. Agarwal and C. S. Burrus, “Fast convolution using Fermat number transforms with 
applications to digital filtering,” IEEE Trans. ASSP, vol. 22, no. 2, pp. 87–97, 1974.
[5] X. Lai and J. L. Massey, “A proposal for a new block encryption standard,” EUROCRYPT ’90, 
LNCS 473, pp. 389–404, 1991.
[6] M. Arshad, MPRC: A Discrete Geometric Theory of Nature, 2025–2026. [Online]. Available: 
https://muhammadarshad.github.io/pages-mprc/
[7] N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed. SIAM, 2002.
[8] W. J. Rider and D. B. Kothe, “Reconstructing volume tracking,” J. Comput. Phys., vol. 141, no. 2,
pp. 112–152, 1998.
[9] G. D. Weymouth and D. K.-P. Yue, “Conservative volume-of-fluid method for free-surface 
simulations on Cartesian grids,” J. Comput. Phys., vol. 229, no. 8, pp. 2853–2865, 2010.
[10] J. Demmel and H. D. Nguyen, “Parallel reproducible summation,” IEEE Trans. Computers, 
vol. 64, no. 7, pp. 2060–2070, 2015.