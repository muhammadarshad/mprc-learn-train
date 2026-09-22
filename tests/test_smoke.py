from mprc_structural.ring import sigma2, delta2

def test_pair():
    assert sigma2(36,25)==61
    assert delta2(36,25)==11
