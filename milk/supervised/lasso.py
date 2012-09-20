# -*- coding: utf-8 -*- 
import numpy as np
import _lasso

def lasso(X, Y, B=None, lam=1., max_iter=None, tol=None):
    '''
    B = lasso(X, Y, B={np.zeros()}, lam=1. max_iter={1024}, tol={1e-15})

    Solve LASSO Optimisation

        B* = arg min_B || Y - BX ||₂² + λ||B||₁

    Parameters
    ----------
    X : ndarray
        Design matrix
    Y : ndarray
        Matrix of outputs
    B : ndarray, optional
        Starting values for approximation. This can be used for a warm start if
        you have an estimate of where the solution should be.
    lam : float, optional
        λ (default: 1.0)
    max_iter : int, optional
        Maximum nr of iterations (default: 1024)
    tol : float, optional
        Tolerance. Used for floating point "equality" comparisions (default: 1e-15)
    '''
    X = np.asfortranarray(X, dtype=np.float32)
    Y = np.asfortranarray(Y, dtype=np.float32)
    if B is None:
        B = np.zeros((Y.shape[0],X.shape[0]), np.float32)
    else:
        B = np.asfortranarray(B, dtype=np.float32)
    if max_iter is None:
        max_iter = 1024
    if tol is None:
        tol = 1e-15
    if X.shape[0] != B.shape[1] or \
        Y.shape[0] != B.shape[0] or \
        X.shape[1] != Y.shape[1]:
        raise ValueError('milk.supervised.lasso: Dimensions do not match')
    _lasso.lasso(X, Y, B, max_iter, float(lam), float(tol))
    return B
