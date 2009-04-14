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

