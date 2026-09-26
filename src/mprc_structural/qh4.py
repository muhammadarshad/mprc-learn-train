"""Exact QH4 active-address map from the frozen Arshad's ViT specification."""

VACUUM = frozenset((0,64,128,192))
GEN = 7
GEN_INV = 183
GATES = 36
STEPS = 7
ACTIVE = 252

def forward(gamma: int, sigma: int) -> int:
    g=int(gamma); s=int(sigma)
    if not 1 <= g <= GATES:
        raise ValueError("gamma must be 1..36")
    if not 1 <= s <= STEPS:
        raise ValueError("sigma must be 1..7")
    return (7*(7*(g-1)+s+(g-1)//9)) & 0xFF

def inverse(p: int) -> tuple[int,int,int]:
    p=int(p)&0xFF
    if p in VACUUM:
        raise ValueError("vacuum has no active QH4 inverse")
    m=(p*GEN_INV)&0xFF
    adj=m-((m-1)//64)
    gamma=(adj-1)//7+1
    sigma=(adj-1)%7+1
    theta=(gamma-1)//9+1
    if forward(gamma,sigma) != p:
        raise ValueError("not an active QH4 address")
    return gamma,sigma,theta

def active_positions() -> tuple[int,...]:
    return tuple(forward(g,s) for g in range(1,37) for s in range(1,8))
