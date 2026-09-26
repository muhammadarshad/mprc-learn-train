"""Paper-3 mixed-radix QH4 geometry and local Arshad Transpose.

Exact source:
    gamma - 1 = 9*theta + 3*a + p
    m = 64*theta + 21*a + 7*p + sigma
    z = 7*m mod 256

theta in Z4, a,p in Z3, sigma in {1,...,7}.
"""

GEN=7
GEN_INV=183
VACUUM=frozenset((0,64,128,192))

def mixed(theta:int,a:int,p:int,sigma:int)->int:
    th=int(theta); aa=int(a); pp=int(p); ss=int(sigma)
    if not 0<=th<4: raise ValueError("theta must be 0..3")
    if not 0<=aa<3 or not 0<=pp<3: raise ValueError("a,p must be 0..2")
    if not 1<=ss<=7: raise ValueError("sigma must be 1..7")
    return 64*th+21*aa+7*pp+ss

def active_address(theta:int,a:int,p:int,sigma:int)->int:
    return (GEN*mixed(theta,a,p,sigma))&0xFF

def gamma(theta:int,a:int,p:int)->int:
    return 1 + 9*int(theta) + 3*int(a) + int(p)

def decode_address(z:int)->tuple[int,int,int,int]:
    zz=int(z)&0xFF
    if zz in VACUUM:
        raise ValueError("vacuum")
    m=(zz*GEN_INV)&0xFF
    theta=m//64
    r=m%64
    if r==0:
        raise ValueError("vacuum offset")
    q=r-1
    a=q//21
    rem=q%21
    p=rem//7
    sigma=rem%7+1
    if active_address(theta,a,p,sigma)!=zz:
        raise ValueError("not active")
    return theta,a,p,sigma

def transpose_local(a:int,p:int,k:int)->tuple[int,int]:
    aa=int(a)%3; pp=int(p)%3; kk=int(k)%3
    return ((pp+kk)%3,(aa-kk)%3)

def sum_coord(a:int,p:int)->int:
    return (int(a)+int(p))%3

def diff_coord(a:int,p:int)->int:
    return (int(p)-int(a))%3
