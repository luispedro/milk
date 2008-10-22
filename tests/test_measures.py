from milk.measures import accuracy
import numpy
from milk.measures import accuracy, waccuracy

def test_100():
    C=numpy.zeros((2,2))
    C[0,0]=100
    C[1,1]=50
    assert accuracy(C) == 1.
    assert waccuracy(C) == 1.

def test_0():
    C = numpy.array([
        [0, 10],
        [10, 0]
        ])
    assert waccuracy(C) == 0.
    assert accuracy(C) == 0.

def test_50():
    C = numpy.array([
        [10, 10],
        [10, 10]
        ])
    assert accuracy(C) == .5
    assert waccuracy(C) == .5

def test_unbalanced():
    C = numpy.array([
        [20, 10],
        [10,  0]
        ])
    assert accuracy(C) == .5
    assert waccuracy(C) == 1./3

