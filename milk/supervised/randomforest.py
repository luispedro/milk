# -*- coding: utf-8 -*-
# Copyright (C) 2010-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

'''
Random Forest
-------------

Main elements
-------------

rf_learner : A learner object
'''

from __future__ import division
import numpy as np
import milk.supervised.tree
from .normalise import normaliselabels
from .base import supervised_model
from ..utils import get_nprandom

__all__ = [
    'rf_learner',
    ]

def _sample(features, labels, n, R):
    '''
    features', labels' = _sample(features, labels, n, R)

    Sample n element from (features,labels)

    Parameters
    ----------
    features : sequence
    labels : sequence
        Same size as labels
    n : integer
    R : random object

    Returns
    -------
    features' : sequence
    labels' : sequence
    '''

    N = len(features)
    sfeatures = []
    slabels = []
    for i in xrange(n):
        idx = R.randint(N)
        sfeatures.append(features[idx])
        slabels.append(labels[idx])
    return np.array(sfeatures), np.array(slabels)

class rf_model(supervised_model):
    def __init__(self, forest, names, return_label = True):
        self.forest = forest
        self.names = names
        self.return_label = return_label

    def apply(self, features):
        rf = len(self.forest)
        votes = sum(t.apply(features) for t in self.forest)
        if self.return_label:
            return (votes > (rf//2))
        return votes / rf


class rf_learner(object):
    '''
    Random Forest Learner

    learner = rf_learner(rf=101, frac=.7)

    Attributes
    ----------
    rf : integer, optional
        Nr of trees to learn (default: 101)
    frac : float, optional
        Sample fraction
    R : np.random object
        Source of randomness
    '''
    def __init__(self, rf=101, frac=.7, R=None):
        self.rf = rf
        self.frac = frac
        self.R = get_nprandom(R)

    def train(self, features, labels, normalisedlabels=False, names=None, return_label=True, **kwargs):
        N,M = features.shape
        m = int(self.frac*M)
        n = int(self.frac*N)
        R = get_nprandom(kwargs.get('R', self.R))
        tree = milk.supervised.tree.tree_learner(return_label=return_label)
        forest = []
        if not normalisedlabels:
            labels,names = normaliselabels(labels)
        elif names is None:
            names = (0,1)
        for i in xrange(self.rf):
            forest.append(
                    tree.train(*_sample(features, labels, n, R),
                               **{'normalisedlabels' : True})) # This syntax is necessary for Python 2.5
        return rf_model(forest, names, return_label)


