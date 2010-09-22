# -*- coding: utf-8 -*-
# Copyright (C) 2010, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from collections import defaultdict
from .classifier import normaliselabels

class voting_classifier(object):
    '''
    Implements a voting scheme for multiple sub-examples per example.
    
    classifier = voting_classifier(base)

    Example
    -------
    
    ::
    
        voterlearn = voting_classifier(milk.supervised.simple_svm())
        voter = voterlearn.train(training_groups,  labeled_groups)
        res = voter.apply([ [f0, f1, f3] ])
    
    '''

    def __init__(self, base):
        self.base = base

    def train(self, gfeatures, glabels):
        if type(gfeatures) == np.ndarray and gfeatures.dtype == object:
            gfeatures = list(gfeatures)
        features = np.concatenate(gfeatures)
        labels = []
        for feats,label in zip(gfeatures, glabels):
            labels.extend( [label] * len(feats) )
        return voting_model(self.base.train(features, labels))


class voting_model(object):
    def __init__(self, base):
        self.base = base

    def apply(self, gfeatures):
        votes = defaultdict(int)
        for feats in gfeatures:
            votes[self.base.apply(feats)] += 1
        best = None
        most_votes = 0
        for k,v in votes.iteritems():
            if v > most_votes:
                best = k
                most_votes = v
        return best


def remove_outliers(features, limit, min_size):
    '''
    features = remove_outliers(features, limit, min_size)

    '''
    nsize = int(limit * len(features))
    if nsize < min_size:
        return features

    normed = features - features.mean(0)
    std = normed.std(0)
    std[std == 0] = 1
    normed /= std
    f2_sum1 = (normed**2).mean(1)
    values = f2_sum1.copy()
    values.sort()
    top = values[nsize]
    selected = f2_sum1 < top
    return features[selected]


class filter_outliers_model(object):
    def __init__(self, limit, min_size):
        self.limit = limit
        self.min_size = min_size

    def apply(self, features):
        return remove_outliers(features, self.limit, self.min_size)

class filter_outliers(object):
    def __init__(self, limit=.9, min_size=3):
        self.limit = limit
        self.min_size = min_size

    def train(self, features, labels):
        return filter_outliers_model(self.limit, self.min_size)

