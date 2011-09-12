import numpy as np
from milk.supervised.precluster import precluster_learner, select_precluster
from milk.tests.fast_classifier import fast_classifier

def c0():
    return np.random.rand(8)
def c1():
    return c0()+2.*np.ones(8)

def gen_data(seed, with_nums=False):
    np.random.seed(seed)

    features = []
    labels =[]
    for i in xrange(200):
        f = []
        for j in xrange(40):
            use_0 = (i < 100 and j < 30) or (i >= 100 and j >= 30)
            if use_0: f.append(c0())
            else: f.append(c1())
        labels.append((i < 100))
        if with_nums:
            features.append((f,[]))
        else:
            features.append(f)
    return features, labels


def test_precluster():
    learner = precluster_learner([2], base=fast_classifier(), R=12)
    features, labels = gen_data(22)
    model = learner.train(features,labels)

    assert model.apply([c0() for i in xrange(35)])
    assert not model.apply([c1() for i in xrange(35)])

def test_codebook_learner():
    learner = select_precluster([2,3,4], base=fast_classifier())
    learner.rmax = 3
    features, labels = gen_data(23, 1)
    model = learner.train(features,labels)

    assert model.apply(([c0() for i in xrange(35)],[]))
    assert not model.apply(([c1() for i in xrange(35)],[]))

def test_codebook_learner_case1():
    learner = select_precluster([2], base=fast_classifier())
    learner.rmax = 1
    features, labels = gen_data(23, 1)
    model = learner.train(features,labels)

    assert model.apply(([c0() for i in xrange(35)],[]))
    assert not model.apply(([c1() for i in xrange(35)],[]))

