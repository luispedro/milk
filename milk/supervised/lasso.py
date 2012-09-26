# -*- coding: utf-8 -*- 
import numpy as np
import _lasso
from .base import supervised_model
from milk.unsupervised import center

def lasso(X, Y, B=None, lam=1., max_iter=None, tol=None):
    '''
    B = lasso(X, Y, B={np.zeros()}, lam=1. max_iter={1024}, tol={1e-5})

    Solve LASSO Optimisation

        B* = arg min_B || Y - BX ||₂² + λ||B||₁

    Milk uses coordinate descent, looping through the coordinates in order
    (with an active set strategy to update only non-zero βs, if possible). The
    problem is convex and the solution is guaranteed to be optimal (within
    floating point accuracy).

    Parameters
    ----------
    X : ndarray
        Design matrix
    Y : ndarray
        Matrix of outputs
    B : ndarray, optional
        Starting values for approximation. This can be used for a warm start if
        you have an estimate of where the solution should be. If used, the
        solution might be written in-place (if the array has the right format).
    lam : float, optional
        λ (default: 1.0)
    max_iter : int, optional
        Maximum nr of iterations (default: 1024)
    tol : float, optional
        Tolerance. Whenever a parameter is to be updated by a value smaller
        than ``tolerance``, that is considered a null update. Be careful that
        if the value is too small, performance will degrade horribly.
        (default: 1e-5)

    Returns
    -------
    B : ndarray
    '''
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    if B is None:
        B = np.zeros((Y.shape[0],X.shape[0]), np.float32)
    else:
        B = np.ascontiguousarray(B, dtype=np.float32)
    if max_iter is None:
        max_iter = 1024
    if tol is None:
        tol = 1e-5
    if X.shape[0] != B.shape[1] or \
        Y.shape[0] != B.shape[0] or \
        X.shape[1] != Y.shape[1]:
        raise ValueError('milk.supervised.lasso: Dimensions do not match')
    if np.any(np.isnan(X)) or np.any(np.isnan(B)):
        raise ValueError('milk.supervised.lasso: NaNs are only supported in the ``Y`` matrix')
    W = np.ascontiguousarray(~np.isnan(B), dtype=np.float32)
    Y = np.nan_to_num(Y)
    _lasso.lasso(X, Y, W, B, max_iter, float(lam), float(tol))
    return B

def lasso_walk(X, Y, B=None, nr_steps=None, start=None, step=None, tol=None):
    '''
    Bs = lasso_walk(X, Y, B={np.zeros()}, nr_steps={64}, start={automatically inferred}, step={.9}, tol=None)

    Repeatedly solve LASSO Optimisation

        B* = arg min_B || Y - BX ||₂² + λ||B||₁

    for different values of λ.

    Parameters
    ----------
    X : ndarray
        Design matrix
    Y : ndarray
        Matrix of outputs
    B : ndarray, optional
        Starting values for approximation. This can be used for a warm start if
        you have an estimate of where the solution should be.
    start : float, optional
        first λ to use (default is ``np.abs(Y).max()``)
    nr_steps : int, optional
        How many steps in the path (default is 64)
    step : float, optional
        Multiplicative step to take (default is 0.9)
    tol : float, optional
        This is the tolerance parameter. It is passed to the lasso function
        unmodified.

    Returns
    -------
    Bs : ndarray
    '''
    if nr_steps is None:
        nr_steps = 64
    if step is None:
        step = .9
    if start is None:
        start = np.nanmax(np.abs(Y))*np.abs(X).max()


    lam = start
    Bs = []
    for i in xrange(nr_steps):
        # The central idea is that each iteration is already "warm" and this
        # should be faster than starting from zero each time
        B = lasso(X, Y, B, lam=lam, tol=tol)
        Bs.append(B.copy())
        lam *= step
    return np.array(Bs)

def _dict_subset(mapping, keys):
    return dict(
            [(k,mapping[k]) for k in keys])

class lasso_model(supervised_model):
    def __init__(self, betas, mean):
        self.betas = betas
        self.mean = mean

    def retrain(self, features, labels, lam, **kwargs):
        features, mean = center(features) 
        betas = lasso(features, labels, self.betas.copy(), lam=lam, **_dict_subset(kwargs, ['tol', 'max_iter']))
        return lasso_model(betas, mean)
        
    def apply(self, features):
        return np.dot(self.betas, features) + self.mean


class lasso_learner(object):
    def __init__(self, lam=1.0):
        self.lam = lam

    def train(self, features, labels, betas=None, **kwargs):
        labels, mean = center(labels, axis=1) 
        betas = lasso(features, labels, betas, lam=self.lam)
        return lasso_model(betas, mean)


