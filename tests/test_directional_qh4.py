from mprc_structural.directional_adi import encode,decode,s5,continuation,transpose
from mprc_structural.qh4 import forward,inverse,active_positions,VACUUM

def test_directional_adi_basis():
    for k in range(9):
        for v in range(256):
            a=[0]*9; a[k]=v
            z=encode(a)
            assert decode(z)==tuple(a)

def test_directional_adi_transpose_involution():
    z=encode((11,29,47,83,109,137,163,211,239))
    assert transpose(transpose(z))==z

def test_qh4_full_active_bijection():
    p=active_positions()
    assert len(p)==252
    assert len(set(p))==252
    assert set(range(256))-set(p)==set(VACUUM)
    for g in range(1,37):
        for s in range(1,8):
            q=forward(g,s)
            gg,ss,theta=inverse(q)
            assert (gg,ss)==(g,s)
