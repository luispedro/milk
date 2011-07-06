import numpy as np
from milk.supervised.precluster_learner import precluster_learner
from milk.tests.fast_classifier import fast_classifier

def test_precluster():
    np.random.seed(22)
    learner = precluster_learner([2], base=fast_classifier(), R=12)

    def c0():
        return np.random.rand(8)
    def c1():
        return c0()+2.*np.ones(8)

    features = []
    labels =[]
    for i in xrange(200):
        f = []
        for j in xrange(40):
            use_0 = (i < 100 and j < 30) or (i >= 100 and j >= 30)
            if use_0: f.append(c0())
            else: f.append(c1())
        labels.append((i < 100))
        features.append(f)
    model = learner.train(features,labels)


    assert model.apply([c0() for i in xrange(35)])
    assert not model.apply([c1() for i in xrange(35)])

