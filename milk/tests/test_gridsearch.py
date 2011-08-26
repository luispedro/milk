import milk.supervised.gridsearch
import milk.supervised.svm
from milk.supervised.gridsearch import gridminimise, _allassignments, gridsearch
from milk.tests.fast_classifier import fast_classifier
from nose.tools import raises
import numpy as np


def slow_gridminimise(learner, features, labels, params, measure=None):
    from ..measures.nfoldcrossvalidation import nfoldcrossvalidation
    if measure is None:
        measure = np.trace

    best_val = initial_value
    best = None
    for assignement in _allassignments(params):
        _set_assignment(learner, assignement)
        S,_ = nfoldcrossvalidation(features, labels, classifier=learner)
        cur = measure(S)
        if cur > best_val:
            best = assignement
            best_val = cur
    return best


def test_gridsearch():
    from milksets import wine
    features, labels = wine.load()
    selected = (labels < 2)
    features = features[selected]
    labels = labels[selected]

    G = milk.supervised.gridsearch(
            milk.supervised.svm.svm_raw(),
            params={'C':[.01,.1,1.,10.],
                    'kernel':[milk.supervised.svm.rbf_kernel(0.1),milk.supervised.svm.rbf_kernel(1.)]
            })
    model = G.train(features,labels)
    reslabels = [model.apply(f) for f in features]
    assert len(reslabels) == len(features)
test_gridsearch.slow = True


def test_all_assignements():
    assert len(list(_allassignments({'C': [0,1], 'kernel' : ['a','b','c']}))) == 2 * 3

class error_learner(object):
    def train(self, features, labels, **kwargs):
        raise ValueError('oops')
    
    def set_option(self, k, v):
        pass

@raises(Exception)
def test_with_error():
    from milksets.wine import load
    features, labels = load()
    learner = error_learner()
    G = milk.supervised.gridsearch(
        error_learner(),
        params = { 'error' : range(3), 'error2' : range(5) }
        )
    G.train(features,labels)
    

class simple_model:
    def __init__(self, c):
        self.c = c
    def apply(self, f):
        return self.c

def f(a,b,c):
    return a**2 + b**3 + c

class simple_learner:
    def set_option(self, k, v):
        setattr(self, k, v)
    def train(self, fs, ls, normalisedlabels=False):
        return simple_model(f(self.a, self.b, self.c))

def test_gridminimise():
    features = np.arange(100)
    labels = np.tile((0,1), 50)
    paramspace = { 'a': np.arange(4), 'b' : np.arange(-3,3), 'c' : np.linspace(2., 10) }
    best,value = gridminimise(simple_learner(), features, labels, paramspace, measure=(lambda _, p: p[0]), return_value=True)
    best = dict(best)
    val = f(best['a'], best['b'], best['c'])
    assert value == val*100
    for a in np.arange(4):
        for b in np.arange(-3,3):
            for c in np.linspace(2., 10):
                assert val <= f(a,b,c)
    gs = gridsearch(simple_learner(), paramspace, measure=(lambda _, p: p[0]), annotate=True)
    model = gs.train(features, labels)
    assert model.value == value
    assert model.arguments == val

def test_gridminimise():
    from milksets.wine import load
    features, labels = load()
    x = gridminimise(milk.supervised.svm_simple(kernel=np.dot, C=2.), features[::2], labels[::2] == 0, {'C' : (0.5,) })
    cval, = x
    assert cval == ('C', .5)

def test_gridminimise_return():
    from milksets.wine import load
    features,labels = load()
    learner = fast_classifier()
    gridminimise(learner, features, labels, { 'ignore' : [0] })
    _,error = gridminimise(learner, features, labels, { 'ignore' : [0] }, return_value=True, nfolds=5)
    cmat,_ = milk.nfoldcrossvalidation(features, labels, learner=learner, nfolds=5)
    assert error == cmat.sum()-cmat.trace()
