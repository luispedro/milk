import numpy as np
from milk.unsupervised import pdist, plike

def test_pdist():
    np.random.seed(222)
    X = np.random.randn(100,23)
    Y = np.random.randn(80,23)
    Dxx = pdist(X)
    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            assert np.allclose(Dxx[i,j], np.sum((X[i]-X[j])**2))

    Dxy = pdist(X,Y)
    for i in xrange(X.shape[0]):
        for j in xrange(Y.shape[1]):
            assert np.allclose(Dxy[i,j], np.sum((X[i]-Y[j])**2))

