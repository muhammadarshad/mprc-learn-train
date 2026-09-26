"""Corrected directional ADI-9 for Arshad's ViT.

Ordered local byte state:
    C,U1,U2,D1,D2,F1,F2,B1,B2

All arithmetic is native Z256.  No Z512 scalar packing is used.
"""

INV9 = 57
ORDER=("C","U1","U2","D1","D2","F1","F2","B1","B2")
OFFSETS=((0,0),(-1,0),(-2,0),(1,0),(2,0),(0,1),(0,2),(0,-1),(0,-2))

def encode(a):
    if len(a)!=9:
        raise ValueError("directional ADI requires 9 bytes")
    x=[int(v)&0xFF for v in a]
    c=x[0]
    return tuple([sum(x)&0xFF]+[((c-v)&0xFF) for v in x[1:]])

def decode(z):
    if len(z)!=9:
        raise ValueError("directional ADI requires Lambda + 8 deltas")
    q=[int(v)&0xFF for v in z]
    c=(INV9*((q[0]+sum(q[1:]))&0xFF))&0xFF
    return tuple([c]+[((c-d)&0xFF) for d in q[1:]])

def s5(z):
    q=tuple(int(v)&0xFF for v in z)
    c=decode(q)[0]
    # first-depth U1,D1,F1,B1 are deltas 1,3,5,7
    return (5*c-q[1]-q[3]-q[5]-q[7])&0xFF

def continuation(z):
    q=tuple(int(v)&0xFF for v in z)
    return (
        (q[1]-q[2])&0xFF,
        (q[3]-q[4])&0xFF,
        (q[5]-q[6])&0xFF,
        (q[7]-q[8])&0xFF,
    )

def transpose(z):
    # spatial transpose: U<->B, D<->F, depths preserved.
    q=tuple(int(v)&0xFF for v in z)
    return (q[0],q[7],q[8],q[5],q[6],q[3],q[4],q[1],q[2])

def extract_2d(image,row,col):
    h=len(image); w=len(image[0])
    r=int(row); c=int(col)
    if r<2 or r>=h-2 or c<2 or c>=w-2:
        raise ValueError("center needs two cells of support on all four arms")
    a=[image[r+dr][c+dc] for dr,dc in OFFSETS]
    return encode(a)
