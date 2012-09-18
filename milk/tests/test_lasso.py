import milk.supervised.lasso
import numpy as np

def test_lasso_smoke():
    np.random.seed(3)
    for i in xrange(8):
        X = np.random.rand(100,10)
        Y = np.random.rand(5,10)
        B = np.random.rand(5,100)
        before = np.linalg.norm(Y - np.dot(B,X))
        B  = milk.supervised.lasso.lasso(X,Y)
        after = np.linalg.norm(Y - np.dot(B,X))
        assert after < before
        assert np.all(~np.isnan(B))
