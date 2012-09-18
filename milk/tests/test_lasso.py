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

def test_lasso_nans():
    np.random.seed(3)
    for i in xrange(8):
        X = np.random.rand(100,10)
        Y = np.random.rand(5,10)
        B = np.random.rand(5,100)
        for j in xrange(12):
            Y.flat[np.random.randint(0,Y.size-1)] = float('nan')
        B  = milk.supervised.lasso.lasso(X,Y)
        assert np.all(~np.isnan(B))

def test_lam_zero():
    np.random.seed(2)
    for i in xrange(8):
        X = np.random.rand(24,2)
        Y = np.random.rand(1,2)
        B  = milk.supervised.lasso.lasso(X,Y, lam=0.0)
        R = Y - np.dot(B,X)
        R = R.ravel()
        assert np.dot(R,R) < .01
