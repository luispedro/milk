import numpy as np
import random
import milk.supervised.svm
import milk.supervised.multi
from milk.supervised.classifier import ctransforms

import milksets
features,labels = milksets.wine.load()
A = np.arange(len(features))
random.seed(9876543210)
random.shuffle(A)
features = features[A]
labels = labels[A]
base = lambda : ctransforms(milk.supervised.svm.svm_raw(C=2.,kernel=milk.supervised.svm.rbf_kernel(2.**-3)),milk.supervised.svm.svm_binary())

def test_one_against_rest():
    M = milk.supervised.multi.one_against_rest(base)
    M = M.train(features[:100,:],labels[:100])
    tlabels = [M.apply(f) for f in features[100:]]
    for tl in tlabels:
        assert tl in (1,2,3)

def test_one_against_one():
    M = milk.supervised.multi.one_against_one(base)
    M = M.train(features[:100,:],labels[:100])
    tlabels = [M.apply(f) for f in features[100:]]
    for tl in tlabels:
        assert tl in (1,2,3)

def test_two_thirds():
    np.random.seed(2345)
    C = milk.supervised.defaultclassifier()
    X = np.random.rand(120,4)
    X[:40] += np.random.rand(40,4)
    X[:40] += np.random.rand(40,4)
    X[40:80] -= np.random.rand(40,4)
    X[40:80] -= np.random.rand(40,4)
    Y = np.repeat(np.arange(3), 40)
    model = C.train(X,Y)
    Y_ = np.array([model.apply(x) for x in X])
    assert (Y_ == Y).mean() * 3 > 2

