import numpy.random
import milk.unsupervised.pca
import numpy as np

def test_pca():
    numpy.random.seed(123)
    X = numpy.random.rand(10,4)
    X[:,1] += numpy.random.rand(10)**2*X[:,0] 
    X[:,1] += numpy.random.rand(10)**2*X[:,0] 
    X[:,2] += numpy.random.rand(10)**2*X[:,0] 
    Y,V = milk.unsupervised.pca(X)
    Xn = milk.unsupervised.normalise.zscore(X)
    assert X.shape == Y.shape
    assert ((np.dot(V[:4].T,Y[:,:4].T).T-Xn)**2).sum()/(Xn**2).sum() < .3

def test_mds():
    from milk.unsupervised import pdist
    np.random.seed(232)
    for _ in xrange(12):
        features = np.random.random_sample((12,4))
        X = milk.unsupervised.mds(features,4)
        D = pdist(features)
        D2 = pdist(X)
        assert np.mean( (D - D2) ** 2) < 10e-4


def test_mds_dists():
    from milk.unsupervised import pdist
    np.random.seed(232)
    for _ in xrange(12):
        features = np.random.random_sample((12,4))
        D = pdist(features)
        X = milk.unsupervised.mds(features,4)
        X2 = milk.unsupervised.mds_dists(D, 4)
        assert np.mean( (X - X2) ** 2) < 10e-4


