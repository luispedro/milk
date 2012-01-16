# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from .classifier import normaliselabels
import multiprocessing

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

def _set_options(learner, options):
    for k,v in options:
        learner.set_option(k,v)

class Grid1(multiprocessing.Process):
    def __init__(self, learner, features, labels, measure, train_kwargs, options, folds, inq, outq):
        self.learner = learner
        self.features = features
        self.labels = labels
        self.measure = measure
        self.train_kwargs = train_kwargs
        self.options = options
        self.folds = folds
        self.inq = inq
        self.outq = outq
        super(Grid1, self).__init__()

    def execute_one(self, index, fold):
        _set_options(self.learner, self.options[index])
        train, test = self.folds[fold]
        model = self.learner.train(self.features[train], self.labels[train], normalisedlabels=True, **self.train_kwargs)
        preds = [model.apply(f) for f in self.features[test]]
        error = self.measure(self.labels[test], preds)
        return error

    def run(self):
        try:
            while True:
                index,fold = self.inq.get()
                if index == 'shutdown':
                    self.outq.close()
                    self.outq.join_thread()
                    return
                error = self.execute_one(index, fold)
                self.outq.put( (index, error) )
        except Exception, e:
            import traceback
            errstr = r'''\
Error in milk.gridminimise internal

Exception was: %s

Original Traceback:
%s

(Since this was run on a different process, this is not a real stack trace).
''' % (e, traceback.format_exc())
            self.outq.put( ('error', errstr) )


def gridminimise(learner, features, labels, params, measure=None, nfolds=10, return_value=False, train_kwargs=None, nprocs=None):
    '''
    best = gridminimise(learner, features, labels, params, measure={0/1 loss}, nfolds=10, return_value=False, nprocs=None)
    best, value = gridminimise(learner, features, labels, params, measure={0/1 loss}, nfolds=10, return_value=True, nprocs=None)

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
    return_value : boolean, optional
        Whether to return the error value as well. Default False
    train_kwargs : dict, optional
        Options that are passed to the train() method of the classifier, using
        the ``train(features, labels, **train_kwargs)`` syntax. Defaults to {}.
    nprocs : integer, optional
        Number of processors to use. By default, uses the
        ``milk.utils.parallel`` framework to check the number of
        processors.

    Returns
    -------
    best : a sequence of assignments
    value : float
        Only returned if ``return_value`` is true
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
    from ..utils import parallel
    if measure is None:
        from ..measures.measures import zero_one_loss
        measure = zero_one_loss
    if train_kwargs is None:
        train_kwargs = {}
    try:
        features = np.asanyarray(features)
    except:
        features = np.array(features, dtype=object)

    labels,_ = normaliselabels(labels)
    options = list(_allassignments(params))
    iteration = np.zeros(len(options), int)
    error = np.zeros(len(options), float)
    folds = [(Tr.copy(), Te.copy()) for Tr,Te in foldgenerator(labels, nfolds)]
    # foldgenerator might actually decide on a smaller number of folds,
    # depending on the distribution of class sizes:
    nfolds = len(folds)
    assert nfolds
    if nprocs is None:
        nprocs = len(options)
    else:
        nprocs = min(nprocs, len(options))
    assert nprocs > 0, 'milk.supervised.gridminimise: nprocs <= 0!!'
    nprocs = parallel.get_procs(nprocs, use_current=True)

    executing = set()
    workers = []
    if nprocs > 1:
        inqueue = multiprocessing.Queue()
        outqueue = multiprocessing.Queue()
        for i in xrange(nprocs):
            inqueue.put((i,0))
            executing.add(i)

            w = Grid1(learner, features, labels, measure, train_kwargs, options, folds, inqueue, outqueue)
            w.start()
            workers.append(w)
        getnext = outqueue.get
        queuejob = lambda next, fold: inqueue.put( (next, fold) )
    else:
        worker = Grid1(learner, features, labels, measure, train_kwargs, options, folds, None, None)
        queue = []
        def queuejob(index,fold):
            queue.append((index,fold))
        def getnext():
            index,fold = queue.pop()
            return index, worker.execute_one(index,fold)
        queuejob(0,0)
        executing.add(0)

    try:
        while True:
            p,err = getnext()
            if p == 'error':
                raise RuntimeError(err)
            executing.remove(p)
            iteration[p] += 1
            error[p] += err
            for best in np.where(error == error.min())[0]:
                if iteration[best] == nfolds:
                    if return_value:
                        return options[best], error[best]
                    return options[best]
            for next in error.argsort():
                if iteration[next] < nfolds and next not in executing:
                    executing.add(next)
                    queuejob(next, iteration[next])
                    break
    finally:
        assert np.max(iteration) <= nfolds
        if len(workers):
            for w in workers:
                inqueue.put( ('shutdown', None) )
            inqueue.close()
            inqueue.join_thread()
            for w in workers:
                w.join()
        parallel.release_procs(nprocs, count_current=True)


class gridsearch(object):
    '''
    G = gridsearch(base, measure=accuracy, nfolds=10, params={ param1 : [...], param2 : [...]}, annotate=False)

    Perform a grid search for the best parameter values.

    When G.train() is called, then for each combination of p1 in param1, p2 in
    param2, ... it performs (effectively)::

        base.param1 = p1
        base.param2 = p2
        ...
        value[p1, p2,...] = measure(nfoldcrossvalidation(..., learner=base))

    it then picks the highest set of parameters and re-learns a model on the
    whole data.

    Parameters
    -----------
    base : classifier to use
    measure : function, optional
        a function that takes labels and outputs and returns the loss.
        Default: 0/1 loss. This must be an *additive* function.
    nfolds : integer, optional
        Nr of folds
    params : dictionary
    annotate : boolean
        Whether to annotate the returned model with ``arguments`` and ``value``
        fields with the result of cross-validation. Defaults to False.

    All of the above can be *passed as parameters to the constructor or set as
    attributes*.

    See Also
    --------
    gridminimise : function
        Implements the basic functionality behind this object
    '''
    def __init__(self, base, measure=None, nfolds=10, params={}, annotate=False):
        self.params = params
        self.base = base
        self.nfolds = 10
        self.measure = measure
        self.annotate = annotate

    def is_multi_class(self):
        return self.base.is_multi_class()

    def train(self, features, labels, normalisedlabels=False, **kwargs):
        best,value = gridminimise(self.base, features, labels, self.params, self.measure, self.nfolds, return_value=True, train_kwargs=kwargs)
        _set_options(self.base, best)
        model = self.base.train(features, labels, normalisedlabels=normalisedlabels, **kwargs)
        if self.annotate:
            model.arguments = best
            model.value = value
        return model

