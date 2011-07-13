# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from .normalise import normaliselabels
from .base import supervised_model

'''
AdaBoost

Simple implementation of Adaboost

Learner
-------

boost_learner

'''

__all__ = [
    'boost_learner',
    ]

def _adaboost(features, labels, base, max_iters):
    m = len(features)
    D = np.ones(m, dtype=float)
    D /= m
    Y = np.ones(len(labels), dtype=float)
    names = np.array([-1, +1])
    Y = names[labels]
    H = []
    A = []
    for t in xrange(max_iters):
        Ht = base.train(features, labels, weights=D)
        train_out = np.array(map(Ht.apply, features))
        train_out = names[train_out]
        Et = np.dot(D, (Y != train_out))
        if Et > .5:
            # early return
            break
        At = .5 * np.log((1. + Et) / (1. - Et))
        D *= np.exp((-At) * Y * train_out)
        D /= np.sum(D)
        A.append(At)
        H.append(Ht)
    return H, A


class boost_model(supervised_model):
    def __init__(self, H, A, names):
        self.H = H
        self.A = A
        self.names = names

    def apply(self, f):
        v = sum((a*h.apply(f)) for h,a in zip(self.H, self.A))
        v /= np.sum(self.A)
        return self.names[v > .5]


class boost_learner(object):
    '''
    learner = boost_learner(weak_learner_type(), max_iters=100)
    model = learner.train(features, labels)
    test = model.apply(f)

    AdaBoost learner

    Attributes
    ----------
    base : learner
        Weak learner
    max_iters : integer
        Nr of iterations (default: 100)
    '''
    def __init__(self, base, max_iters=100):
        self.base = base
        self.max_iters = max_iters

    def train(self, features, labels, normalisedlabels=False, names=(0,1), weights=None, **kwargs):
        if not normalisedlabels:
            labels,names = normaliselabels(labels)
        H,A = _adaboost(features, labels, self.base, self.max_iters)
        return boost_model(H, A, names)
