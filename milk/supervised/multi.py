# -*- coding: utf-8 -*-
# Copyright (C) 2008, Luís Pedro Coelho <lpc@cmu.edu>
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

    base must be a callable object that provides the base classifier to use.

    Example
    -------

        multi = one_against_rest(milk.supervised.simple_svm)
        multi.train(training_features,labels)
        print multi.apply(testing_features)

    We are using a class as a base, but we can use any callable object:

        multi = one_against_rest(lambda : milk.supervised.tree(purity_measure='gini'))
    ...

    See Also
    -------
    one_against_one
    '''

    def __init__(self,base):
        self.classifiers = None
        self.base = base
        self.is_multi_class = True
        self.trained = False
        self.options = {}

    def set_option(self, k, v):
        self.options[k] = v

    def train(self,features,labels):
        labels, self.names = normaliselabels(labels)
        self.nclasses = labels.max() + 1
        self.classifiers = []
        for i in xrange(self.nclasses):
            s = self.base()
            for k,v in self.options.iteritems():
                s.set_option(k, v)
            s.train(features, labels == i)
            self.classifiers.append(s)
        self.trained = True

    def apply(self,feats):
        assert self.trained
        vals = np.array([c.apply(feats) for c in self.classifiers])
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

    base must be a callable object that provides the base classifier to use.

    Example
    -------

        multi = one_against_one(milk.supervised.simple_svm)
        multi.train(training_features,labels)
        print multi.apply(testing_features)

    We are using a class as a base, but we can use any callable object:

        multi = one_against_one(lambda : milk.supervised.tree(purity_measure='gini'))
    ...

    See Also
    -------
    one_against_rest
    '''

    def __init__(self, base):
        self.classifiers = None
        self.base = base
        self.is_multi_class = True
        self.trained = False
        self.options = {}

    def set_option(self, k, v):
        self.options[k] = v

    def train(self, features, labels):
        '''
        one_against_one.train(objs,labels)
        '''
        labels, self.names = normaliselabels(labels)
        self.nclasses = labels.max() + 1
        self.classifiers = [ [None for i in xrange(self.nclasses)] for j in xrange(self.nclasses)]
        for i in xrange(self.nclasses):
            for j in xrange(i+1,self.nclasses):
                s = self.base()
                for k,v in self.options.iteritems():
                    s.set_option(k, v)
                idxs = (labels == i) | (labels == j)
                assert idxs.sum() > 0, 'milk.multi.one_against_one: Pair-wise classifier has no data'
                # Fixme: here I could add a Null classifier or something
                s.train(features[idxs],labels[idxs]==i)
                self.classifiers[i][j] = s
        self.trained = True

    def apply(self,feats):
        '''
        one_against_one.apply(objs)

        Classify one single object.
        '''
        assert self.trained
        nc = self.nclasses
        votes = np.zeros(nc)
        for i in xrange(nc):
            for j in xrange(i+1,nc):
                c = self.classifiers[i][j].apply(feats)
                if c:
                    votes[i] += 1
                else:
                    votes[j] += 1
        return self.names[votes.argmax(0)]
    
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
