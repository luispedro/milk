import milk.supervised.tree
import milk.supervised._tree
from milk.supervised._tree import set_entropy
from milk.supervised.tree import information_gain, stump_learner
import numpy as np

def test_tree():
    from milksets import wine
    features, labels = wine.load()
    selected = (labels < 2)
    features = features[selected]
    labels = labels[selected]
    C = milk.supervised.tree.tree_classifier()
    model = C.train(features,labels)
    assert (np.array([model.apply(f) for f in features]) == labels).mean() > .5


def test_split_subsample():
    import random
    from milksets import wine
    features, labels = wine.load()
    labels = labels.astype(np.int)

    seen = set()
    for i in xrange(20):
        random.seed(2)
        i,s = milk.supervised.tree._split(features[::10], labels[::10], None, milk.supervised.tree.information_gain, 2, random)
        seen.add(i)
    assert len(seen) <= 2


def test_set_entropy():
    labels = np.arange(101)%3
    counts = np.zeros(3)
    entropy = milk.supervised._tree.set_entropy(labels, counts)
    slow_counts = np.array([(labels == i).sum() for i in xrange(3)])
    assert np.all(counts == slow_counts)
    px = slow_counts.astype(float)/ slow_counts.sum()
    slow_entropy = - np.sum(px * np.log(px))
    assert np.abs(slow_entropy - entropy) < 1.e-8


def slow_information_gain(labels0, labels1):
    H = 0.
    N = len(labels0) + len(labels1)
    nlabels = 1+max(labels0.max(), labels1.max())
    counts = np.empty(nlabels, np.double)
    for arg in (labels0, labels1):
        H -= len(arg)/float(N) * set_entropy(arg, counts)
    return H

def test_information_gain():
    np.random.seed(22)
    for i in xrange(8):
        labels0 = (np.random.randn(20) > .2).astype(int)
        labels1 = (np.random.randn(33) > .8).astype(int)
        fast = information_gain(labels0, labels1)
        slow = slow_information_gain(labels0, labels1)
        assert np.abs(fast - slow) < 1.e-8


def test_information_gain_small():
    labels1 = np.array([0])
    labels0 = np.array([0, 1])
    assert information_gain(labels0, labels1) < 0.


def test_z1_loss():
    from milk.supervised.tree import z1_loss
    L0 = np.zeros(10)
    L1 = np.ones(10)
    L1[3] = 0
    W0 = np.ones(10)
    W1 = np.ones(10)
    assert z1_loss(L0, L1) == z1_loss(L0, L1, W0, W1)
    assert z1_loss(L0, L1) != z1_loss(L0, L1, W0, .8*W1)
    assert z1_loss(L0, L1) > 0


def test_stump_learner():
    learner = stump_learner()
    np.random.seed(111)
    for i in xrange(8):
        features = np.random.random_sample((40,2))
        features[:20,0] += .5
        labels = np.repeat((0,1),20)
        model = learner.train(features, labels, normalisedlabels=True)
        assert not model.apply([0.01,.5])
        assert model.apply(np.random.random_sample(2)+.8)
        assert model.idx == 0

