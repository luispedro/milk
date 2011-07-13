# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from .normalise import normaliselabels
from .base import supervised_model

__all__ = [
    'logistic_learner',
    ]

@np.vectorize
def _sigmoidal(z):
    if (z > 300): return 1.
    if z < -300: return 0.
    return 1./(1+np.exp(-z))

class logistic_model(supervised_model):
    def __init__(self, bs):
        self.bs = bs

    def apply(self, fs):
        return _sigmoidal(self.bs[0] + np.dot(fs, self.bs[1:]))

class logistic_learner(object):
    '''
    learner = logistic_learner(alpha=0.0)

    Logistic regression learner

    There are two implementations:

    1. One which depends on ``scipy.optimize``. This is the default and is
       extremely fast.
    2. If ``import scipy`` fails, then we fall back to a Python only
       gradient-descent. This gives good results, but is many times slower.

    Properties
    ----------

    alpha : real, optional
        penalty for L2-normalisation. Default is zero, for no penalty.

    '''
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def train(self, features, labels, normalisedlabels=False, names=None, **kwargs):
        def error(bs):
            response = bs[0] + np.dot(features, bs[1:])
            response = _sigmoidal(response)
            diff = response - labels
            log_like = np.dot(diff, diff)
            L2_penalty = self.alpha * np.dot(bs, bs)
            return log_like + L2_penalty
        def error_prime(bs):
            fB = np.dot(features, bs[1:])
            response = _sigmoidal(bs[0] + fB)
            sprime = response * (1-response)
            ds = (response - labels) * sprime
            b0p = np.sum(ds)
            b1p = np.dot(features.T, ds)
            bp = np.concatenate( ([b0p], b1p) )
            return 2.*(bp + self.alpha*bs)

        features = np.asanyarray(features)
        if not normalisedlabels:
            labels, _ = normaliselabels(labels)
        N,f = features.shape
        bs = np.zeros(f+1)
        try:
            from scipy import optimize
            # Some testing revealed that this was a good combination
            # call fmin_cg twice first and then fmin
            # I do not understand why 100%, but there it is
            bs = optimize.fmin_cg(error, bs, error_prime, disp=False)
            bs = optimize.fmin_cg(error, bs, error_prime, disp=False)
            bs = optimize.fmin(error, bs, disp=False)
        except ImportError:
            import warnings
            warnings.warn('''\
milk.supervised.logistic.train: Could not import scipy.optimize.
Fall back to very simple gradient descent (which is slow).''')
            bs = np.zeros(f+1)
            cur = 1.e-6
            ebs = error(bs)
            for i in xrange(1000000):
                dir = error_prime(bs)
                step = (lambda e : bs - e *dir)
                enbs = ebs + 1
                while enbs > ebs:
                    cur /= 2.
                    if cur == 0.:
                        break
                    nbs = step(cur)
                    enbs = error(nbs)
                while cur < 10.:
                    cur *= 2
                    nnbs = step(cur)
                    ennbs = error(nnbs)
                    if ennbs < enbs:
                        nbs = nnbs
                        enbs = ennbs
                    else:
                        break
                bs = nbs
                ebs = enbs
        return logistic_model(bs)
