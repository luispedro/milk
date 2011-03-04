import milk.supervised.svm
import milk.supervised.normalise
import numpy as np
import milk.supervised.svm

def _slow_f(features,labels,kernel_or_sigma):
    try:
        kernel = kernel_or_sigma
        kernel(features[0],features[1])
    except:
        kernel = milk.supervised.svm.rbf_kernel(kernel_or_sigma)
    N1 = (labels == 0).sum()
    N2 = (labels == 1).sum()
    x1 = features[labels == 0]
    x2 = features[labels == 1]
    dm = 0
    for i in xrange(N1):
        for j in xrange(N1):
            dm += kernel(x1[i],x1[j])/N1/N1
    for i in xrange(N2):
        for j in xrange(N2):
            dm += kernel(x2[i],x2[j])/N2/N2
    for i in xrange(N1):
        for j in xrange(N2):
            dm -= 2*kernel(x1[i],x2[j])/N1/N2
    s1 = N1
    for i in xrange(N1):
        for j in xrange(N1):
            s1 -= kernel(x1[i],x1[j])/N1
    s2 = N2
    for i in xrange(N2):
        for j in xrange(N2):
            s2 -= kernel(x2[i],x2[j])/N2
    return (s1 + s2)/dm


def test_fisher_approx():
    from milksets import wine
    features,labels = wine.load()
    f = milk.supervised.svm.sigma_value_fisher(features,labels)
    for sigma in (2.**-4,2.,16.,32.):
        assert abs(f(sigma) - _slow_f(features,labels,sigma)) < 1e-6
