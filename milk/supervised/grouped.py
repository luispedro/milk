# -*- coding: utf-8 -*-
# Copyright (C) 2010, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from collections import defaultdict

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
        features = np.concatenate(gfeatures)
        labels = []
        for feats,label in zip(gfeatures, glabels):
            labels.extend( [label] * len(feats) )
        return voting_model(self.base.train(features, labels))


class voting_model(object):
    def __init__(self, base):
        self.base = base

    def apply(self, gfeatures):
        res = []
        for features in gfeatures:
            votes = defaultdict(int)
            for feats in features:
                votes[self.base.apply(feats)] += 1
            best = None
            most_votes = 0
            for k,v in votes.iteritems():
                if v > most_votes:
                    best = k
                    most_votes = v
            res.append(best)
        return res

