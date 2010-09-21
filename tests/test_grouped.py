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


base = lambda: ctransforms(milk.supervised.svm.svm_raw(C=2.,kernel=milk.supervised.svm.rbf_kernel(2.**-3)),milk.supervised.svm.svm_binary())
base = milk.supervised.multi.one_against_rest(base)
features,labels = milksets.wine.load()
gfeatures, glabels = group(features, labels, 3)


def test_voting():
    learner = milk.supervised.grouped.voting_classifier(base)
    learner.train(gfeatures, glabels)
    model = learner.train(gfeatures, glabels)
    assert len(model.apply(gfeatures)) == len(glabels)
    assert (model.apply(gfeatures) == np.array(glabels)).mean() > .8

