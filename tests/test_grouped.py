import numpy as np
import milk.supervised.svm
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
    base = lambda: ctransforms(milk.supervised.svm.svm_raw(C=2.,kernel=milk.supervised.svm.rbf_kernel(2.**-3)),milk.supervised.svm.svm_binary())
    base = milk.supervised.multi.one_against_rest(base)
    features,labels = milksets.wine.load()
    gfeatures, glabels = group(features, labels, 3)

    learner = milk.supervised.grouped.voting_classifier(base)
    learner.train(gfeatures, glabels)
    model = learner.train(gfeatures, glabels)
    assert len(model.apply(gfeatures)) == len(glabels)
    assert (model.apply(gfeatures) == np.array(glabels)).mean() > .8


def test_filter_outliers():
    np.random.seed(22)
    features = [np.random.randn(10,10) for i in xrange(20)]
    for f in features:
        f[0] *= 10
        

    trainer = milk.supervised.grouped.filter_outliers(.9)
    model = trainer.train(features, [0] * len(features))
    for ff,f in zip(model.apply(features), features):
        assert np.all(ff == f[1:])

