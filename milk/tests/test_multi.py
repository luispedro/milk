import numpy as np
import random
import milk.supervised.svm
import milk.supervised.multi
from milk.supervised.classifier import ctransforms
from fast_classifier import fast_classifier

import milksets.wine
features,labels = milksets.wine.load()
A = np.arange(len(features))
random.seed(9876543210)
random.shuffle(A)
features = features[A]
labels = labels[A]
labelset = set(labels)
base = ctransforms(milk.supervised.svm.svm_raw(C=2.,kernel=milk.supervised.svm.rbf_kernel(2.**-3)),milk.supervised.svm.svm_binary())

def test_one_against_rest():
    M = milk.supervised.multi.one_against_rest(base)
    M = M.train(features[:100,:],labels[:100])
    tlabels = [M.apply(f) for f in features[100:]]
    for tl in tlabels:
        assert tl in labelset

def test_one_against_one():
    M = milk.supervised.multi.one_against_one(base)
    M = M.train(features[:100,:],labels[:100])
    tlabels = [M.apply(f) for f in features[100:]]
    for tl in tlabels:
        assert tl in labelset

def test_two_thirds():
    np.random.seed(2345)
    C = milk.supervised.defaultclassifier('fast')
    X = np.random.rand(120,4)
    X[:40] += np.random.rand(40,4)
    X[:40] += np.random.rand(40,4)
    X[40:80] -= np.random.rand(40,4)
    X[40:80] -= np.random.rand(40,4)
    Y = np.repeat(np.arange(3), 40)
    model = C.train(X,Y)
    Y_ = np.array([model.apply(x) for x in X])
    assert (Y_ == Y).mean() * 3 > 2

def test_multi_labels():
    clabels = [[lab, lab+7] for lab in labels]
    multi_label = milk.supervised.multi.one_against_rest_multi(base)
    model = multi_label.train(features[::2], clabels[::2])
    test_vals = [model.apply(f) for f in features[1::2]]
    for ts in test_vals:
        if 0.0 in ts: assert 7.0 in ts
        if 1.0 in ts: assert 8.0 in ts
        if 2.0 in ts: assert 9.0 in ts


def test_classifier_no_set_options():
    # Basically these should not raise an exception
    milk.supervised.multi.one_against_rest_multi(fast_classifier())
    milk.supervised.multi.one_against_rest(fast_classifier())
    milk.supervised.multi.one_against_one(fast_classifier())

