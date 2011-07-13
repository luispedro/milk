# -*- coding: utf-8 -*-
# Copyright (C) 2010-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from collections import defaultdict
from .classifier import normaliselabels
from .base import base_adaptor, supervised_model

__all__ = [
    'voting_learner',
    'mean_learner',
    'remove_outliers',
    'filter_outliers',
    ]

def _concatenate_features_labels(gfeatures, glabels):
        if type(gfeatures) == np.ndarray and gfeatures.dtype == object:
            gfeatures = list(gfeatures)
        features = np.concatenate(gfeatures)
        labels = []
        for feats,label in zip(gfeatures, glabels):
            labels.extend( [label] * len(feats) )
        return features, labels

class voting_learner(base_adaptor):
    '''
    Implements a voting scheme for multiple sub-examples per example.

    classifier = voting_learner(base)

    base should be a binary classifier

    Example
    -------

    ::

        voterlearn = voting_learner(milk.supervised.simple_svm())
        voter = voterlearn.train(training_groups,  labeled_groups)
        res = voter.apply([ [f0, f1, f3] ])

    '''

    def train(self, gfeatures, glabels, normalisedlabels=False):
        features, labels = _concatenate_features_labels(gfeatures, glabels)
        return voting_model(self.base.train(features, labels))
voting_classifier = voting_learner


class voting_model(supervised_model):
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

class mean_learner(base_adaptor):
    '''
    Implements a mean scheme for multiple sub-examples per example.

    classifier = mean_learner(base)

    `base` should be a classifier that returns a numeric confidence value
    `classifier` will return the **mean**

    Example
    -------

    ::

        meanlearner = mean_learner(milk.supervised.raw_svm())
        model = meanlearner.train(training_groups,  labeled_groups)
        res = model.apply([ [f0, f1, f3] ])

    '''
    def train(self, gfeatures, glabels, normalisedlabels=False):
        features, labels = _concatenate_features_labels(gfeatures, glabels)
        return mean_model(self.base.train(features, labels))

mean_classifier = mean_learner

class mean_model(supervised_model):
    def __init__(self, base):
        self.base = base

    def apply(self, gfeatures):
        return np.mean([self.base.apply(feats) for feats in gfeatures])


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


class filter_outliers_model(supervised_model):
    def __init__(self, limit, min_size):
        self.limit = limit
        self.min_size = min_size

    def apply(self, features):
        return remove_outliers(features, self.limit, self.min_size)

class filter_outliers(object):
    def __init__(self, limit=.9, min_size=3):
        self.limit = limit
        self.min_size = min_size

    def train(self, features, labels, normalisedlabels=False):
        return filter_outliers_model(self.limit, self.min_size)

