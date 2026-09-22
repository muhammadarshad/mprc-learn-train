# Arshad's ViT v1 coding specification

> Text extract of the author-supplied source document used for this research repo.

<PARSED TEXT FOR PAGE: 1 / 5>
Arshad's ViT
Self-Contained Coding Specification v1.0
Frozen QCM / QH4 / ADI / GEVHV vision architecture on Z256
0. Purpose and coding contract
This document is sufficient to implement Arshad's ViT v1 without consulting the author's wider research. Do not import 
Swin/ViT attention semantics, FFT features, floating-point trigonometry, arbitrary Even x Even byte windows, or 
undocumented MPRC conventions.
The goal is to implement and test a candidate replacement for Swin-style local/shifted-window attention. Structural 
mathematics below is frozen. Performance superiority is not assumed; it must be measured. Unfrozen semantics, 
especially the internal 17-bit metadata codec and learned LUT policy, remain explicit interfaces.
1. Frozen constants and substrate
Value Definition / role
R=Z256 native byte ring; uint8 storage
tau=256 full turn; reduce with &0xFF
iota=+64 quarter-turn
H x W=128 x 113 QCM manifold; 14,464 states
V={0,64,128,192} vacuum/frame boundaries
ACTIVE=252 36 gates x 7 steps
GEN=7 2^0+2^1+2^2; gcd(7,256)=1
GEN_INV=183 7^-1 mod 256
GATES=36 4 x 9
STEPS=7 sigma per gate
QUARTERS=4 theta domains
EPOCH=9 3 x 3 gates per quarter
SIEVE=16 7+9=7+2+7
252 = 36*7 = 4*9*7 = 4*(3*3)*7
256 = 4*(7*9 + 1)
Seven is the generator. Do not replace it with 8.
2. QH4 address
Active address: Lambda(rho,gamma,sigma,theta), with rho=winding/amplitude, gamma=1..36, sigma=1..7, theta=1..4.
p = v mod 256
rho = floor(v/256)
Verified forward map:
pos(gamma,sigma) = (7*(7*(gamma-1) + sigma + floor((gamma-1)/9))) mod 256
Verified inverse for active p:
m=(p*183) mod 256
adj=m-floor((m-1)/64)
gamma=floor((adj-1)/7)+1
sigma=((adj-1) mod 7)+1
theta=floor((gamma-1)/9)+1
Gate: exact on all 252 active positions, zero collisions.
<PARSED TEXT FOR PAGE: 2 / 5>
3. Axis semantics
Item Frozen meaning
X / Movement straight traversal; gamma/gate
Y / Position straight position; sigma/step
Z / Charge structural circle/depth; modular horizon; not Ring1 phase 
amplitude
U / Rotation CW/CCW from change of phi; never stored as position
128 antipodal singularity/direction flip
U is undefined at vacuums. Do not append it as an arbitrary learned positional channel.
4. Image geometry versus byte topology
Images may be square. Byte-computational structures must not be forced into Even x Even windows.
112 = 16*7
Native byte rectangles are 16x7 and 7x16. The 64 vacuum-spacing/address domain is NOT an 8x8 byte grid.
128*113=14464
112*112=12544
14464-12544=1920=15*128=(4^2-1)*128
Preserve the 112x112 payload exactly. Reserve 1,920 states as structural/header capacity. The 17-bit header mapping 
is not frozen here; expose a MetadataCodec interface.
5. Local attention geometry
k=3
k^2=9=1+8
neighbors=k^2-1=8
Nine is the 3x3 gate structure per quarter: one reference/accumulation plus eight differential relations, not a learned 9x9
attention matrix.
6. ADI local representation
Lambda=sum(a_i), i=1..N
delta_k=a_1-a_(k+1), k=1..N-1
a_1=(Lambda+sum(delta_k))/N
a_(k+1)=a_1-delta_k
For local vision attention N=9. Ring differences use unsigned modular subtraction. Retain k^2-1 as the relation count.
7. Generator interaction invariants
g=[1,2,4]
sum(g)=7
g dot g=1+4+16=21
g outer g =
[1 2 4]
[2 4 8]
[4 8 16]
sum=49
16=7+9=7+2+7. These are structural diagnostics, not trainable constants.
<PARSED TEXT FOR PAGE: 3 / 5>
8. GEVHV attention
IDENTIFY -> BIND -> REACT -> MEASURE
8.1 BIND
phi_(s,t)(g)=s*g+t mod 256, s odd
phi^-1(y)=s^-1*(y-t) mod 256
phi_v(g)=g+v mod 256
8.2 REACT
S5(g)[i,j]=g[i,j]+g[i-1,j]+g[i+1,j]+g[i,j-1]+g[i,j+1] mod 256
rho(g)[i,j]=LUT[S5(g)[i,j]]
Boundary is identity in v1. Fusion:
Lprime[u]=L[(s*u+5*t) mod 256]
S5(g+v)=S5(g)+S5(v) mod 256
c=S5(v)
rho(g+v)[i,j]=L[(S5(g)[i,j]+c[i,j]) mod 256]
8.3 MEASURE
cdist(a,b)=min((a-b) mod 256,(b-a) mod 256)
E(g,v)=sum_sites cdist(react(bind(g,v)),v)
Energy uses a wide ordinary integer and is never folded. Full-grid max: 1,851,392 < 2^31.
9. Seven-step propagation
support(r)=1+4*(1+...+r)=1+2*r*(r+1)
support(7)=113
Required topology test: seven 5-site reaction rounds produce exact 113-site Manhattan support for an interior impulse.
10. Ring/hardware rules
 Ring states are uint8; reduce with &0xFF.
 Wrapping s32/u32 accumulation then &0xFF is legal for ring GEMM because 256 divides 2^32.
 Signed and unsigned byte lanes are congruent modulo 256.
 No floating point required in the ring attention path.
 No NumPy dependency in the reference implementation.
 Native integer multiply or Quarter-Square LUT are allowed only if bit-exact.
 cdist uses two modular subtractions plus unsigned min.
11. Reference forward pass
1 validate image bytes
2 losslessly pack payload into 128x113 manifold
3 fill structural region only via MetadataCodec
4 derive QH4 addresses
5 identify 3x3 as 1 reference + 8 ADI relations
6 transport with generator-7/QH4 navigation; no 8x8 byte windows
7 use 16x7 <-> 7x16 transpose for row/column phases
8 BIND query hypervector
9 REACT with 5-site LUT; canonical topology test = 7 rounds
10 MEASURE circular-distance energy
11 return energy, diagnostics, optional reacted manifold
Batch energies are independent. Parallelize across manifolds; never construct an N x N attention matrix.
<PARSED TEXT FOR PAGE: 4 / 5>
12. Multi-resolution policy
Required tests: 24x24, 112x112, 220x220, 320x320. Preserve every source byte. Do not resize merely to fit 
conventional patches and do not introduce Even x Even byte windows. Use 16x7 / 7x16 structures plus QH4 
addressing. Non-closing dimensions require explicit padding/partial-structure metadata; padding is not a vacuum state. 
Optimal non-112 tiling remains experimental. The prior 8x8/14x14 interpretation is rejected.
13. Verification gates
Gate Pass criterion
G1 QH4 252/252 exact; zero collisions
G2 generator 7*183 mod 256=1; full permutation
G3 ADI exact round-trip, including randomized N=9 
neighborhoods
G4 cdist all 65,536 byte pairs exact
G5 scalar bind all odd s, all t, all bytes invert
G6 fusion materialized vs LUT-absorbed bit-exact
G7 vector bind materialized g+v vs offset field bit-exact
G8 packing 112x112 payload exact round-trip
G9 propagation 7 rounds => 113-site support
G10 transpose 16x7 -> 7x16 -> 16x7 exact
G11 multires 24/112/220/320 preserve all source bytes; no Even x 
Even byte windows
G12 optimization fast path bit-exact before timing
14. Evidence ladder
Layer Question Evidence
A Algebra Is construction correct? G1-G10 exact tests
B Implementation Does code equal spec? reference vs optimized bit-exact
C Connectivity Does information escape local 
structures?
influence graph / impulse coverage
D Learning Can it learn vision representations? synthetic spatial tasks, then small 
image benchmark
E Comparison Can it replace Swin competitively? matched accuracy, convergence, 
memory, ops, latency
Do not change frozen constants to repair a learning/benchmark failure. Diagnose operator, metadata codec, 
LUT/training rule, head, or routing policy.
15. Minimal software architecture
arshad_vit/
 z256.py
 qh4.py
 adi.py
 topology.py
 manifold.py
 bind.py
 react.py
 measure.py
 model.py
 multires.py
 tests/test_qh4.py
 tests/test_adi.py
 tests/test_ring.py
 tests/test_bind.py
 tests/test_react.py
 tests/test_manifold.py
 tests/test_topology.py
<PARSED TEXT FOR PAGE: 5 / 5>
 tests/test_multires.py
 tests/test_reference_vs_fast.py
16. Required public interfaces
class MetadataCodec:
 def encode(self,image,structural_region): ...
 def decode(self,manifold): ...
class ReactionLUT:
 def table(self) -> bytes: ... # exactly 256 outputs
class ArshadsViT:
 def forward(self,image_u8,query_u8,metadata_codec,lut,rounds=7,return_manifold=False): ...
If real metadata/LUT is absent, use explicitly named NeutralMetadataCodec and IdentityLUT. Such runs are 
topology/execution tests, not recognition results.
17. Non-negotiable prohibitions
 No 8x8 or other Even x Even byte attention windows.
 No silent resizing to fit conventional ViT patches.
 No hidden standard Q/K/V + softmax reference path.
 No FFT replacement.
 No float sine/cosine in the ring path.
 No signed state representation as fundamental storage.
 No invented 17-bit metadata semantics.
 No changing frozen constants 7,9,16,36,64,112,113,128,252,256 for benchmarks.
 No performance claim before exactness gates pass.
18. Coding-agent completion definition
Milestone 1 is complete only when G1-G12 pass; all four required resolutions execute without losing source bytes; 
connectivity reporting shows influence growth across local structures/orientation phases; scalar reference and optimized
paths are bit-exact; and one reproducible command emits a machine-readable verification report. Only then add a 
classifier/training head and compare with Swin.
19. Frozen summary
SPACE : Z256, vacuums {0,64,128,192}
ADDRESS : Lambda(rho,gamma,sigma,theta)
ACTIVE : 252=36*7=4*9*7
LOCAL : 9=3*3=1 reference + 8 ADI relations
GENERATOR : 7, inverse 183
BYTE VIEW : 16x7 <-> 7x16; never forced Even x Even
MANIFOLD : 128x113=14464
PAYLOAD : 112x112=12544
STRUCTURAL : 1920=15*128
ATTENTION : IDENTIFY -> BIND -> REACT -> MEASURE
REACTION : 5-site cross LUT on Z256
CANONICAL : 7 rounds -> 113-site support
READOUT : integer circular-distance energy
END OF FROZEN CODING SPECIFICATION v1.0