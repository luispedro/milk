# -*- coding: utf-8 -*-
# Copyright (C) 2011, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution
'''
============
Jug Parallel
============

These are some functions that make it easier to take advantage of `jug
<http://luispedro.org/software/jug>`__ to perform tasks in parallel.
'''

from __future__ import division
import numpy as np
import milk
try:
    from jug.utils import identity
    from jug.mapreduce import mapreduce
except ImportError:
    raise ImportError('milk.ext.jugparallel requires jug (http://luispedro.org/software/jug')

class _nfold_one(object):
    def __init__(self, features, labels, kwargs):
        self.features = features
        self.labels = labels
        self.kwargs = kwargs

    def __call__(self, i):
        return milk.nfoldcrossvalidation(
                    self.features,
                    self.labels,
                    folds=[i],
                    **self.kwargs)

def _nfold_reduce(a,b):
    cmat = a[0] + b[0]
    names = a[1]
    assert a[1] == b[1]
    if len(a) == 2:
        return cmat, names
    predictions = np.array([a[2],b[2]])
    return cmat, names, predictions.max(0)

def nfoldcrossvalidation(features, labels, **kwargs):
    mapper = identity(_nfold_one(features, labels, kwargs))
    nfolds = kwargs.get('nfolds', 10)
    return mapreduce(_nfold_reduce, mapper, range(nfolds), map_step=1, reduce_step=(nfolds+1))


