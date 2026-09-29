import itertools
import numpy as np
from mprc_structural.parallel_projection import DIRECTIONS,parallel_resolve

def resolver(a,b):
    return ((a+b)&255)

def test_parallel_projection_is_schedule_independent():
    x=np.asarray([1,3,7,15,31,63,127,255],dtype=np.uint8)
    ref=parallel_resolve(x,resolver)
    for p in itertools.permutations(DIRECTIONS):
        got=parallel_resolve(x,resolver,p)
        assert np.array_equal(got["state"],ref["state"])
        for d in DIRECTIONS:
            assert np.array_equal(got["proposals"][d],ref["proposals"][d])

def test_input_is_not_mutated():
    x=np.arange(17,dtype=np.uint8)
    before=x.copy()
    parallel_resolve(x,resolver)
    assert np.array_equal(x,before)
