# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
from collections import defaultdict
import numpy as np

__all__ [
    'kNN',
    ]

class kNN(object):
    '''
    k-Nearest Neighbour Classifier

    Naive implementation of a k-nearest neighbour classifier.

    C = kNN(k)

    Attributes:
    -----------
        * k: the k to use
    '''


    def __init__(self, k=1):
        self.k = k

    def train(self, features, labels, normalisedlabels=False, copy_features=False):
        features = np.asanyarray(features)
        labels = np.asanyarray(labels)
        if copy_features:
            features = features.copy()
            labels = labels.copy()
        return kNN_model(self.k, features, labels)

class kNN_model(object):
    def __init__(self, k, features, labels):
        self.k = k
        self.features = features
        self.labels = labels

    def apply(self,features):
        diff2 = ( (self.features - features)**2 ).sum(1)
        neighbours = diff2.argsort()[:self.k]
        labels = self.labels[neighbours]
        votes = defaultdict(int)
        for L in labels:
            votes[L] += 1
        v,L = max( (v,L) for L,v in votes.items() )
        return L

