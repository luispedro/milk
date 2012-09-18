import numpy as np
import _lasso

def lasso(X, Y, B=None, lam=1., max_iter=None, eps=None):
    X = np.asfortranarray(X, dtype=np.float32)
    Y = np.asfortranarray(Y, dtype=np.float32)
    if B is None:
        B = np.zeros((Y.shape[0],X.shape[0]), np.float32)
    else:
        B = np.asfortranarray(B, dtype=np.float32)
    if max_iter is None:
        max_iter = 1024
    if eps is None:
        eps = 1e-15
    if X.shape[0] != B.shape[1] or \
        Y.shape[0] != B.shape[0] or \
        X.shape[1] != Y.shape[1]:
        raise ValueError('milk.supervised.lasso: Dimensions do not match')
    _lasso.lasso(X, Y, B, max_iter, float(lam), float(eps))
    return B
