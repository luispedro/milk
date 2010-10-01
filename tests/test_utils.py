import numpy as np
from milk.utils.utils import get_nprandom, get_pyrandom

def test_nprandom():
    assert get_nprandom(None).rand() != get_nprandom(None).rand()
    assert get_nprandom(1).rand() != get_nprandom(2).rand()
    assert get_nprandom(1).rand() == get_nprandom(1).rand()
    r =  get_nprandom(1)
    assert get_nprandom(r).rand() != r.rand()

def test_pyrandom():
    assert get_pyrandom(None).random() != get_pyrandom(None).random()
    assert get_pyrandom(1).random() != get_pyrandom(2).random()
    assert get_pyrandom(1).random() == get_pyrandom(1).random()
    r =  get_pyrandom(1)
    assert get_pyrandom(r).random() != r.random()

def test_cross_random():
    assert get_pyrandom(get_nprandom(1)).random() == get_pyrandom(get_nprandom(1)).random()
    assert get_nprandom(get_pyrandom(1)).rand() == get_nprandom(get_pyrandom(1)).rand()

def test_recursive():
    def recurse(f):
        R = f(None)
        assert f(R) is R
    yield recurse, get_pyrandom
    yield recurse, get_nprandom

