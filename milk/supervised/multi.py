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
from .classifier import normaliselabels
import numpy as np
import random

class one_against_rest(object):
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

    def __init__(self,base):
        self.base = base
        self.is_multi_class = True
        self.options = {}

    def set_option(self, k, v):
        self.options[k] = v

    def train(self,features,labels):
        labels, names = normaliselabels(labels)
        nclasses = labels.max() + 1
        models  = []
        for i in xrange(nclasses):
            for k,v in self.options.iteritems():
                self.base.set_option(k, v)
            model = self.base.train(features, labels == i)
            models.append(model)
        return one_against_rest_model(models, names)

class one_against_rest_model(object):
    def __init__(self, models, names):
        self.models = models
        self.nclasses = len(self.models)
        self.names = names

    def apply(self, feats):
        vals = np.array([c.apply(feats) for c in self.models])
        idxs, = np.where(vals == 1)
        if len(idxs) == 1:
            label = idxs[0]
        elif len(idxs) == 0:
            label = random.randint(0, self.nclasses - 1)
        else:
            label = random.choice(idxs)
        return self.names[label]


class one_against_one(object):
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


    def __init__(self, base):
        self.base = base
        self.is_multi_class = True
        self.options = {}


    def set_option(self, k, v):
        self.options[k] = v

    def train(self, features, labels):
        '''
        one_against_one.train(objs,labels)
        '''
        labels, names = normaliselabels(labels)
        nclasses = labels.max() + 1
        models = [ [None for i in xrange(nclasses)] for j in xrange(nclasses)]
        for i in xrange(nclasses):
            for j in xrange(i+1, nclasses):
                for k,v in self.options.iteritems():
                    self.base.set_option(k, v)
                idxs = (labels == i) | (labels == j)
                assert idxs.sum() > 0, 'milk.multi.one_against_one: Pair-wise classifier has no data'
                # Fixme: here I could add a Null model or something
                model = self.base.train(features[idxs],labels[idxs]==i)
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

class one_against_rest_multi(object):
    '''
    learner = one_against_rest_multi()
    model = learner.train(features, labels)
    classes = model.apply(f_test)
    '''

    def __init__(self, base):
        self.base = base
        self.set_option = base.set_option

    def train(self, features, labels):
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

