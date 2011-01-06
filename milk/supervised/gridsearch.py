# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from .classifier import normaliselabels

__all__ = [
    'gridminimise',
    'gridsearch',
    ]

def _allassignments(options):
    try:
        from itertools import product
    except ImportError:
        def product(*args, **kwds):
            # from http://docs.python.org/library/itertools.html#itertools.product
            pools = map(tuple, args) * kwds.get('repeat', 1)
            result = [[]]
            for pool in pools:
                result = [x+[y] for x in result for y in pool]
            for prod in result:
                yield tuple(prod)
    from itertools import repeat, izip
    for ks,vs in izip(repeat(options.keys()), product(*options.values())):
        yield zip(ks,vs)

def _set_assignment(obj,assignments):
    for k,v in assignments:
        obj.set_option(k,v)

def gridminimise(learner, features, labels, params, measure=None, nfolds=10):
    '''
    best = gridminimise(learner, features, labels, params, measure={0/1 loss})

    Grid search for the settings of parameters that maximises a given measure

    This function is equivalent to searching the grid, but does not actually
    search the whole grid.

    Parameters
    ----------
    learner : a classifier object
    features : sequence of features
    labels : sequence of labels
    params : dictionary of sequences
        keys are the options to change,
        values are sequences of corresponding elements to try
    measure : function, optional
        a function that takes labels and outputs and returns the loss.
        Default: 0/1 loss. This must be an *additive* function.
    nfolds : integer, optional
        nr of folds to run, default: 10

    Returns
    -------
    best : a sequence of assignments
    '''
    # The algorithm is as follows:
    #
    # for all assignments: error = 0, next_iteration = 0
    #
    # at each iteration:
    #    look for assignment with smallest error
    #    if that is done: return it
    #    else: perform one more iteration
    #
    # When the function returns, that assignment has the lowest error of all
    # assignments and all the iterations are done. Therefore, other assignments
    # could only be worse even if we never computed the whole error!

    from ..measures.nfoldcrossvalidation import foldgenerator
    if measure is None:
        def measure(real, preds):
            return np.sum(np.asarray(real) != np.asarray(preds))

    labels,_ = normaliselabels(labels)
    allassignments = list(_allassignments(params))
    N = len(allassignments)
    iteration = np.zeros(N, int)
    error = np.zeros(N, float)
    folds = [(Tr.copy(), Te.copy()) for Tr,Te in foldgenerator(labels, nfolds)]
    while True:
        next_pos = (error == error.min())
        iter = iteration[next_pos].max()
        if iter == nfolds:
            (besti,) = np.where(next_pos & (iteration == iter))
            besti = besti[0]
            return allassignments[besti]
        (ps,) = np.where(next_pos & (iteration == iter))
        p = ps[0]
        _set_assignment(learner, allassignments[p])
        train, test = folds[iter]
        model = learner.train(features[train], labels[train], normalisedlabels=True)
        preds = [model.apply(f) for f in features[test]]
        error[p] += measure(labels[test], preds)
        iteration[p] += 1


class gridsearch(object):
    '''
    G = gridsearch(base, measure=accuracy, nfolds=10, params={ 'param1 : [...], param2 : [...]})

    Perform a grid search for the best parameter values.


    When G.train() is called, then for each combination of p1 in param1, p2 in
    param2, ... it performs::

        base_classifier.param1 = p1
        base_classifier.param2 = p2
        ...
        value[p1, p2,...] = measure(crossvaliation(base_classifier)

    it then picks the highest set of parameters and re-learns a model on the
    whole data.


    Parameters
    -----------
    base_classifier : classifier to use
    measure : function, optional
        a function that takes labels and outputs and returns the loss.
        Default: 0/1 loss. This must be an *additive* function.
    nfolds : integer, optional
        Nr of folds
    params : dictionary
    '''
    def __init__(self, base, measure=None, nfolds=10, params={}):
        self.params = params
        self.base = base
        self.nfolds = 10
        self.measure = measure

    def is_multi_class(self):
        return self.base.is_multi_class()

    def train(self, features, labels, normalisedlabels=False):
        self.best = gridminimise(self.base, features, labels, self.params, self.measure, self.nfolds)
        _set_assignment(self.base, self.best)
        return self.base.train(features, labels, normalisedlabels=normalisedlabels)

