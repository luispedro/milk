# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from __future__ import division
import numpy as np

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
        yield izip(ks,vs)

def _set_assignment(obj,assignments):
    for k,v in assignments:
        obj.set_option(k,v)

def gridmaximise(learner, features, labels, params, measure=None, initial_value=-1):
    '''
    best = gridmaximise(learner, features, labels, params, measure={accuracy} initial_value=-1)

    Grid search for the settings of parameters that maximises a given measure

    Parameters
    ----------
    learner : a classifier object
    features : sequence of features
    labels : sequence of labels
    params : dictionary of sequences
        keys are the options to change,
        values are sequences of corresponding elements to try
    measure : function, optional
        This function should take a confusion matrix and return a measure of how good it is.
        By default, measure accuracy
    initial_value : any, optional

    Returns
    -------
    best : a sequence of assignments
    '''
    from ..measures.nfoldcrossvalidation import nfoldcrossvalidation
    if measure is None:
        measure = np.trace

    best_val = initial_value
    best = None
    for assignement in _allassignments(params):
        _set_assignment(learner, assignement)
        S,_ = nfoldcrossvalidation(features, labels, classifier=learner)
        cur = measure(S)
        if cur > best_val:
            best = assignement
            best_val = cur
    return best


class gridsearch(object):
    '''
    G = gridsearch(base, measure=accuracy, param1=[...], param2=[...], ...)

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
      measure : a function which takes a confusion matrix and outputs
                 how good the matrix is
    '''
    def __init__(self, base, measure=None, params={}):
        self.params = params
        self.base = base
        self.best = None
        self.measure = measure

    def is_multi_class(self):
        return self.base.is_multi_class()

    def train(self,features,labels):
        self.best = gridmaximise(self.base, features, labels, self.params, self.measure)
        _set_assignment(self.base, self.best)
        return self.base.train(features, labels)

