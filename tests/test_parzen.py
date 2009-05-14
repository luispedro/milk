from __future__ import division
import milk.supervised.normalise
from milk.supervised.parzen import get_parzen_rbf_loocv
import numpy as np
import milksets

def _slow_parzen(features, labels, sigma):
    correct = 0
    N = len(features)
    labels = 2*labels - 1
    def kernel(fi, fj):
        return np.exp(-((fi-fj)**2).sum()/sigma)
    for i in xrange(N):
        C = 0.
        for j in xrange(N):
            if i == j: continue
            C += labels[j] * kernel(features[i],features[j])
        if (C*labels[i] > 0): correct += 1
    return correct/N

def test_parzen():
    features,labels = milksets.wine.load()
    labels = (labels == 1)
    features = milk.supervised.normalise.zscore(features)
    f = get_parzen_rbf_loocv(features, labels)
    sigmas = 2.**np.arange(-4,4)
    for s in sigmas:
        assert abs(_slow_parzen(features, labels, s) - f(s)) < 1e-6
