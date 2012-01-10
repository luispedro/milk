# -*- coding: utf-8 -*-
# Copyright (C) 2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution
'''
============
Jug Parallel
============

These are some functions that make it easier to take advantage of `jug
<http://luispedro.org/software/jug>`__ to perform tasks in parallel.

All of the functions in this module return a jug.Task object and are usable
with the ``CompoundTask`` interface in jug.

None of this will make sense unless you understand and have used jug.
'''

from __future__ import division
import numpy as np
import milk
try:
    from jug import TaskGenerator, value
    from jug.utils import identity
    from jug.mapreduce import mapreduce
    from jug.mapreduce import reduce as jug_reduce
except ImportError:
    raise ImportError('milk.ext.jugparallel requires jug (http://luispedro.org/software/jug)')

def _nfold_reduce(a,b):
    cmat = a[0] + b[0]
    names = a[1]
    assert a[1] == b[1]
    if len(a) == 2:
        return cmat, names
    predictions = np.array([a[2],b[2]])
    return cmat, names, predictions.max(0)

def nfoldcrossvalidation(features, labels, **kwargs):
    '''
    jug_task = nfoldcrossvalidation(features, labels, **kwargs)

    A jug Task that perform n-foldcrossvalidation

    N-fold cross validation is inherently parallel. This function returns a
    ``jug.Task`` which performs n-fold crossvalidation which jug can
    parallelise.

    Parameters
    ----------
    features : sequence of features
    labels : sequence
    kwargs : any
        This will be passed down to ``milk.nfoldcrossvalidation``

    Returns
    -------
    jug_task : a jug.Task
        A Task object

    See Also
    --------
    milk.nfoldcrossvalidation : The same functionality as a "normal" function
    jug.CompoundTask : This function can be used as argument to CompoundTask
    '''
    nfolds = kwargs.get('nfolds', 10)
    features,labels = map(identity, (features,labels))
    kwargs = {k:identity(v) for k,v in kwargs.iteritems()}
    nfold_one = TaskGenerator(milk.nfoldcrossvalidation)
    mapped = [nfold_one(features, labels, folds=[i], **kwargs) for i in xrange(nfolds)]
    return jug_reduce(_nfold_reduce, mapped)


def _select_min(s0, s1):
    if s0[0] < s1[0]: return s0
    else: return s1

def _evaluate_solution(args):
    features, results, method = args
    from milk.unsupervised.gaussianmixture import AIC, BIC
    if method == 'AIC':
        method = AIC
    elif method == 'BIC':
        method = BIC
    else:
        raise ValueError('milk.ext.jugparallel.kmeans_select_best: unknown method: %s' % method)
    assignments, centroids = results
    value = method(features, assignments, centroids)
    return value, results

def _select_best(features, results, method):
    features = identity(features)
    return mapreduce(_select_min, _evaluate_solution, [(features,r,method) for r in results], reduce_step=32, map_step=8)

def kmeans_select_best(features, ks, repeats=1, method='AIC', R=None, **kwargs):
    '''
    assignments_centroids = kmeans_select_best(features, ks, repeats=1, method='AIC', R=None, **kwargs)

    Perform ``repeats`` calls to ``kmeans`` for each ``k`` in ``ks``, select
    the best one according to ``method.``

    Note that, unlike a raw ``kmeans`` call, this is *always deterministic*
    even if ``R=None`` (which is interpreted as being equivalent to setting it
    to a fixed value). Otherwise, the jug paradigm would be broken as different
    runs would give different results.

    Parameters
    ----------
    features : array-like
        2D array
    ks : sequence of integers
        These will be the values of ``k`` to try
    repeats : integer, optional
        How many times to attempt each k (default: 1).
    method : str, optional
        Which method to use. Must be one of 'AIC' (default) or 'BIC'.
    R : random number source, optional
        Even you do not pass a value, the result will be deterministic. This is
        different from the typical behaviour of ``R``, but, when using jug,
        reproducibility is often but, when using jug, reproducibility is often
        a desired feature.
    kwargs : other options
        These are passed transparently to ``kmeans``

    Returns
    -------
    assignments_centroids : jug.Task
        jug.Task which is the result of the best (as measured by ``method``)
        kmeans clustering.
    '''
    from milk import kmeans
    from milk.utils import get_pyrandom
    kmeans = TaskGenerator(kmeans)
    if R is not None:
        start = get_pyrandom(R).randint(0,1024*1024)
    else:
        start = 7
    results = []
    for ki,k in enumerate(ks):
        for i in xrange(repeats):
            results.append(kmeans(features, k, R=(start+7*repeats*ki+i), **kwargs))
    return _select_best(features, results, method)[1]

