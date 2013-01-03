import milk.measures.measures
import milk.measures.curves
import numpy as np
import numpy
from milk.measures import accuracy, waccuracy, bayesian_significance

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



def test_confusion_matrix():
    np.random.seed(323)
    labels0 = np.arange(101)%3
    labels1 = (labels0 + np.random.rand(101)*2).astype(np.int) % 3
    cmat = milk.measures.measures.confusion_matrix(labels0, labels1)
    for i in xrange(3):
        for j in xrange(3):
            assert cmat[i,j] == np.sum( (labels0 == i) & (labels1 == j) )



def test_significance():
    assert np.allclose(.5, [bayesian_significance(1024,i,i) for i in xrange(0, 1025, 3)])


def test_roc():
    np.random.seed(3)
    for i in xrange(4):
        labels = np.repeat([False,True], 50)
        response = labels + np.random.random(100)*i
        P,R = milk.measures.curves.roc(response, labels != 0)
        assert P.min() >= 0.
        assert R.min() >= 0.
        assert P.max() <= 1.
        assert R.max() <= 1.
