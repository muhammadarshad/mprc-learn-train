from mprc_structural.qh4_mixed import (
    VACUUM, mixed, active_address, decode_address,
    transpose_local, sum_coord, diff_coord
)

def test_mixed_radix_252_exact():
    seen=set()
    for th in range(4):
        local=[]
        for a in range(3):
            for p in range(3):
                for s in range(1,8):
                    m=mixed(th,a,p,s)
                    local.append(m)
                    z=active_address(th,a,p,s)
                    assert z not in seen
                    seen.add(z)
                    assert decode_address(z)==(th,a,p,s)
        assert local==list(range(64*th+1,64*th+64))
    assert len(seen)==252
    assert set(range(256))-seen==set(VACUUM)

def test_local_transpose_theorem_all_states():
    for k in range(3):
        fixed=0
        two_cycle_points=0
        for a in range(3):
            for p in range(3):
                ap=transpose_local(a,p,k)
                assert transpose_local(*ap,k)==(a,p)
                assert sum_coord(*ap)==sum_coord(a,p)
                j=diff_coord(a,p)
                jp=diff_coord(*ap)
                assert jp == (-(j+2*k))%3
                if ap==(a,p):
                    fixed+=1
                    assert j==(-k)%3
                else:
                    two_cycle_points+=1
        assert fixed==3
        assert two_cycle_points==6

def test_global_transpose_orbits_252():
    for k in range(3):
        fixed=0
        visited=set()
        two_cycles=0
        for th in range(4):
            for s in range(1,8):
                for a in range(3):
                    for p in range(3):
                        key=(th,a,p,s)
                        if key in visited:
                            continue
                        a2,p2=transpose_local(a,p,k)
                        key2=(th,a2,p2,s)
                        if key2==key:
                            fixed+=1
                            visited.add(key)
                        else:
                            two_cycles+=1
                            visited.add(key); visited.add(key2)
        assert fixed==84
        assert two_cycles==84
        assert len(visited)==252
