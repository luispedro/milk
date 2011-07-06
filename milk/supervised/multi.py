# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
from .classifier import normaliselabels
from .base import supervised_model, base_adaptor
import numpy as np

__all__ = [
    'one_against_rest',
    'one_against_one',
    'one_against_rest_multi',
    'ecoc_learner',
    ]

def _asanyarray(f):
    try:
        return np.asanyarray(f)
    except:
        return np.array(f, dtype=object)

class one_against_rest(base_adaptor):
    '''
    Implements one vs. rest classification strategy to transform
    a binary classifier into a multi-class classifier.

    classifier = one_against_rest(base)

    base must obey the classifier interface

    Example
    -------

    ::

        multi = one_against_rest(milk.supervised.simple_svm())
        model = multi.train(training_features,labels)
        print model.apply(testing_features)


    See Also
    --------
    one_against_one
    '''


    def train(self, features, labels, normalisedlabels=False):
        labels, names = normaliselabels(labels)
        nclasses = labels.max() + 1
        models  = []
        for i in xrange(nclasses):
            model = self.base.train(features, (labels == i).astype(int), normalisedlabels=True)
            models.append(model)
        return one_against_rest_model(models, names)

class one_against_rest_model(object):
    def __init__(self, models, names):
        self.models = models
        self.nclasses = len(self.models)
        self.names = names

    def apply(self, feats):
        vals = np.array([c.apply(feats) for c in self.models])
        (idxs,) = np.where(vals)
        if len(idxs) == 1:
            (label,) = idxs
        elif len(idxs) == 0:
            label = 0
        else:
            label = idxs[0]
        return self.names[label]


class one_against_one(base_adaptor):
    '''
    Implements one vs. one classification strategy to transform
    a binary classifier into a multi-class classifier.

    classifier = one_against_one(base)

    base must obey the classifier interface

    Example
    -------
    ::

        multi = one_against_one(milk.supervised.simple_svm())
        multi.train(training_features,labels)
        print multi.apply(testing_features)



    See Also
    --------
    one_against_rest
    '''

    def train(self, features, labels, **kwargs):
        '''
        one_against_one.train(objs,labels)
        '''
        labels, names = normaliselabels(labels)
        features = _asanyarray(features)
        nclasses = labels.max() + 1
        models = [ [None for i in xrange(nclasses)] for j in xrange(nclasses)]
        for i in xrange(nclasses):
            for j in xrange(i+1, nclasses):
                idxs = (labels == i) | (labels == j)
                assert idxs.sum() > 0, 'milk.multi.one_against_one: Pair-wise classifier has no data'
                # Fixme: here I could add a Null model or something
                model = self.base.train(features[idxs], (labels[idxs]==i).astype(int), normalisedlabels=True)
                models[i][j] = model
        return one_against_one_model(models, names)


class one_against_one_model(object):
    def __init__(self, models, names):
        self.models = models
        self.names = names
        self.nclasses = len(models)

    def apply(self,feats):
        '''
        one_against_one.apply(objs)

        Classify one single object.
        '''
        nc = self.nclasses
        votes = np.zeros(nc)
        for i in xrange(nc):
            for j in xrange(i+1,nc):
                c = self.models[i][j].apply(feats)
                if c:
                    votes[i] += 1
                else:
                    votes[j] += 1
        return self.names[votes.argmax(0)]

class one_against_rest_multi_model(object):
    def __init__(self, models):
        self.models = models

    def apply(self, feats):
        return [lab for lab,model in self.models.iteritems() if model.apply(feats)]

class one_against_rest_multi(base_adaptor):
    '''
    learner = one_against_rest_multi()
    model = learner.train(features, labels)
    classes = model.apply(f_test)

    This for multi-label problem (i.e., each instance can have more than one label).

    '''
    def train(self, features, labels, normalisedlabels=False):
        '''
        '''
        import operator
        all_labels = set()
        for ls in labels:
            all_labels.update(ls)
        models = {}
        for label in all_labels:
            models[label] = self.base.train(features, [(label in ls) for ls in labels])
        return one_against_rest_multi_model(models)

class ecoc_model(supervised_model):
    def __init__(self, models, codes):
        self.models = models
        self.codes = codes

    def apply(self, f):
        word = np.array([model.apply(f) for model in self.models], bool)
        errors = (self.codes != word).sum(1)
        return np.argmin(errors)
        

class ecoc_learner(base_adaptor):
    '''
    Implements error-correcting output codes for reducing a multi-class problem
    to a set of binary problems.

    Reference
    ---------
    "Solving Multiclass Learning Problems via Error-Correcting Output Codes" by
    T. G. Dietterich, G. Bakiri in Journal of Artificial Intelligence
    Research, Vol 2, (1995), 263-286
    '''

    def train(self, features, labels, normalisedlabels=False, **kwargs):
        if normalisedlabels:
            labelset = np.unique(labels)
        else:
            labels,names = normaliselabels(labels)
            labelset = np.arange(len(names))

        k = len(labelset)
        n = 2**(k-1)
        codes = np.zeros((k,n),bool)
        for k_ in xrange(1,k):
            codes[k_].reshape( (-1, 2**(k-k_-1)) )[::2] = 1
        codes = ~codes
        models = []
        for code in codes.T:
            nlabels = np.zeros(len(labels), bool)
            for ell,c in enumerate(code):
                if c:
                    nlabels[labels == ell] = 1
            models.append(self.base.train(features, nlabels, normalisedlabels=True, **kwargs))
        return ecoc_model(models, codes)

