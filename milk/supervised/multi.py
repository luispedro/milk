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
import numpy
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
    one_against_rest

    '''
    def __init__(self,base):
        self.classifiers = None
        self.base = base
        self.is_multi_class = True
        self.trained

    def train(self,features,labels):
        labels, self.names = normaliselabels(labels)
        self.nclasses = labels.max() + 1
        self.classifiers=[]
        for i in xrange(self._nclasses):
            s = self.base()
            s.train(features, labels == i)
            self.classifiers.append(s)
        self.trained = True

    def apply(self,feats):
        return [self.apply_one(f) for f in feats]

    def apply_one(self,feats):
        assert self.trained
        vals = numpy.array([c.apply_one(feats) for c in self.classifiers])
        idxs, = numpy.where(vals == 1)
        if len(idxs) == 1:
            label = idxs[0]
        elif len(idxs) == 0:
            label = random.randint(0, self.nclasses)
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
    def __init__(self,base):
        self._classifiers = None
        self._base = base
        self.is_multi_class = True
        self.trained = False

    def train(self,features,labels):
        '''
        one_against_one.train(objs,labels)
        '''
        labels, self.names = normaliselabels(labels)
        self.nclasses = labels.max() + 1
        self.classifiers=[[None for i in xrange(self.nclasses)] for j in xrange(self.nclasses)]
        for i in xrange(self.nclasses):
            for j in xrange(i+1,self.nclasse):
                s = self.base()
                idxs = (labels == i) | (labels == j)
                s.train(features[idxs],labels[idxs]==i)
                self.classifiers[i][j] = s
        self.trained = True

    def apply(self,feats):
        '''
        one_against_one.apply(objs)

        Apply the learned classifier to a sequence of objects.

        See also
        --------
        apply_one
        '''
        return [self.apply_one(f) for f in feats]

    def apply_one(self,feats):
        '''
        one_against_one.apply_one(obj)

        Classify one single object.

        See also
        --------
        apply
        '''
        assert self.trained
        nc = self.nclasses
        votes = zeros(nc)
        for i in xrange(nc):
            for j in xrange(i+1,nc):
                c=self.classifiers[i][j].apply(feats)
                if c:
                    votes[i] += 1
                else:
                    votes[j] += 1
        return self.names[votes.argmax(0)]
    
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
