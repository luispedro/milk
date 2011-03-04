import numpy as np
import milk.supervised.svm
from milk.supervised.svm import rbf_kernel
import milk.supervised.multi
import milk.supervised.grouped
from milk.supervised.classifier import ctransforms
import milksets.wine

def group(features, labels, step):
    N = len(labels)
    i = 0
    gfeatures = []
    glabels = []
    while i < N:
        next = i + step
        while next > N or labels[next-1] != labels[i]: next -= 1
        gfeatures.append(features[i:next])
        glabels.append(labels[i])
        i = next
    return gfeatures, glabels



def test_voting():
    base = ctransforms(milk.supervised.svm.svm_raw(C=2.,kernel=milk.supervised.svm.rbf_kernel(2.**-3)),milk.supervised.svm.svm_binary())
    base = milk.supervised.multi.one_against_rest(base)
    features,labels = milksets.wine.load()
    gfeatures, glabels = group(features, labels, 3)

    learner = milk.supervised.grouped.voting_classifier(base)
    learner.train(gfeatures, glabels)
    model = learner.train(gfeatures, glabels)
    assert ([model.apply(f) for f in gfeatures] == np.array(glabels)).mean() > .8


def test_filter_outliers():
    np.random.seed(22)
    features = [np.random.randn(10,10) for i in xrange(20)]
    for f in features:
        f[0] *= 10
        
    trainer = milk.supervised.grouped.filter_outliers(.9)
    model = trainer.train(features, [0] * len(features))
    for f in features:
        ff = model.apply(f)
        assert np.all(ff == f[1:])



def test_nfoldcrossvalidation():
    np.random.seed(22)
    features = np.array([np.random.rand(8+(i%3), 12)*(i//20) for i in xrange(40)], dtype=object)
    labels = np.zeros(40, int)
    labels[20:] = 1
    classifier = milk.supervised.grouped.voting_classifier(milk.supervised.svm_simple(C=1., kernel=rbf_kernel(1./12)))
    cmat, names = milk.nfoldcrossvalidation(features, labels, classifier=classifier)
    assert cmat.shape == (2,2)
    assert sorted(names) == range(2)



class identity_classifier(object):
    def train(self, features, labels):
        return identity_model()

class identity_model(object):
    def apply(self, f):
        return f
    

def test_meanclassif():
    gfeatures = [np.arange(10), np.arange(10)%2]
    glabels = [0,1]
    meanclassif = milk.supervised.grouped.mean_classifier(identity_classifier())
    model = meanclassif.train(gfeatures, glabels)
    assert model.apply(gfeatures[0]) == np.arange(10).mean()
    assert model.apply(gfeatures[1]) == .5

