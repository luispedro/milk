import numpy as np
from milk.unsupervised import pdist, plike

def test_pdist():
    np.random.seed(222)
    X = np.random.randn(100,23)
    Y = np.random.randn(80,23)
    Dxx = pdist(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            assert np.allclose(Dxx[i,j], np.sum((X[i]-X[j])**2))

    Dxy = pdist(X,Y)
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            assert np.allclose(Dxy[i,j], np.sum((X[i]-Y[j])**2))
    Dxye = pdist(X, Y, 'euclidean')
    assert np.allclose(Dxye, np.sqrt(Dxy))

def test_plike():
    np.random.seed(222)
    X = np.random.randn(100,23)
    Lxx = plike(X)
    assert len(Lxx) == len(Lxx.T)
    Lxx2 = plike(X, sigma2=.001)
    assert Lxx[0,1] != Lxx2[0,1]
    assert Lxx[0,0] == Lxx2[0,0]
