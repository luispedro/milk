# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
from collections import defaultdict
import numpy as np
from .base import supervised_model

__all__ = [
    'kNN',
    ]

class kNN(object):
    '''
    k-Nearest Neighbour Classifier

    Naive implementation of a k-nearest neighbour classifier.

    C = kNN(k)

    Attributes:
    -----------
    k : integer
        number of neighbours to consider
    '''


    def __init__(self, k=1):
        self.k = k

    def train(self, features, labels, normalisedlabels=False, copy_features=False):
        features = np.asanyarray(features)
        labels = np.asanyarray(labels)
        if copy_features:
            features = features.copy()
            labels = labels.copy()
        features2 = np.sum(features**2, axis=1)
        return kNN_model(self.k, features, features2, labels)

class kNN_model(supervised_model):
    def __init__(self, k, features, features2, labels):
        self.k = k
        self.features = features
        self.f2 = features2
        self.labels = labels

    def apply(self, features):
        features = np.asanyarray(features)
        diff2 = np.dot(self.features, (-2.)*features)
        diff2 += self.f2
        neighbours = diff2.argsort()[:self.k]
        labels = self.labels[neighbours]
        votes = defaultdict(int)
        for L in labels:
            votes[L] += 1
        v,L = max( (v,L) for L,v in votes.items() )
        return L

