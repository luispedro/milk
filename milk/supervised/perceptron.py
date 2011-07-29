# -*- coding: utf-8 -*-
# Copyright (C) 2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

import numpy as np
from .classifier import normaliselabels
from .base import supervised_model
from . import _perceptron

class perceptron_model(supervised_model):
    def __init__(self, w):
        self.w = w

    def apply(self, f):
        f = np.asanyarray(f)
        v = self.w[0] + np.dot(f, self.w[1:])
        return v > 0

class perceptron_learner(object):
    def __init__(self, eta=.1, max_iters=128):
        self.eta = eta
        self.max_iters = max_iters

    def train(self, features, labels, normalisedlabels=False, **kwargs):
        if not normalisedlabels:
            labels, _ = normaliselabels(labels)
        features = np.asanyarray(features)
        if features.dtype not in (np.float32, np.float64):
            features = features.astype(np.float64)
        weights = np.zeros(features.shape[1]+1, features.dtype)
        for i in xrange(self.max_iters):
            errors = _perceptron.perceptron(features, labels, weights, self.eta)
            if not errors:
                break
        return perceptron_model(weights)


