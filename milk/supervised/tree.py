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
'''
================
Tree Classifier
================

Decision tree based classifier
------------------------------

'''

from __future__ import division
import numpy as np
from collections import defaultdict

class _Leaf(object):
    def __init__(self, v, s):
        self.v = v/float(s)
        self.s = s
    def __repr__(self):
        return 'Leaf(%s,%s)' % (self.v, self.s)

def _split(features, labels, criterion):
    N,f = features.shape
    best = None
    best_val = -1.
    for i in xrange(f):
        domain_i = np.unique(features[:,i])
        for d in domain_i[1:]:
            cur_split = (features[:,i] < d)
            value = criterion(labels[cur_split],labels[~cur_split])
            if value > best_val:
                best_val = value
                best = i,d
    return best


def _entropy_set(labels):
    from scipy.stats import entropy
    counts = defaultdict(float)
    for l in counts:
        counts[l] += 1.
    return entropy(counts.values())


def information_gain(*args,**kwargs):
    '''
    ig = information_gain(labels0, labels1, ..., include_entropy=False)

    Information Gain
    See http://en.wikipedia.org/wiki/Information_gain_in_decision_trees

    The function calculated here does not include the original entropy unless
    you explicitly ask for it (by passing include_entropy=True)
    '''
    assert len(args) > 1, 'information_gain: only defined for two or more labels'
    H = 0.
    N = sum(len(A) for A in args)
    if kwargs.get('include_entropy',False):
        counts = defaultdict(float)
        for arg in args:
            for L in arg:
                counts[L] += 1.
        H = scipy.stats.entropy(counts.values())
    for arg in args:
        H -= len(arg)/N * _entropy_set(arg)
    return H        


def build_tree(features, labels, criterion, min_split=4):
    '''
    tree = build_tree(features, labels, criterion, min_split=4)

    Parameters
    ----------
        * features: features to use
        * labels: labels
        * criterion: function to measure goodness of split
        * min_split: minimum size to split on
    '''
    assert len(features) == len(labels), 'build_tree: Nr of labels does not match nr of features'
    features = np.asanyarray(features)
    labels = np.asanyarray(labels)
    if len(labels) < min_split:
        return _Leaf(labels.sum(), float(len(labels)))
    S = _split(features, labels, criterion)
    if S is None:
        return _Leaf(labels.sum(), float(len(labels)))
    i,thresh = S
    split = features[:,i] < thresh
    data0 = features[split]
    data1 = features[~split]
    labels0 = labels[split]
    labels1 = labels[~split]
    return (i, thresh, build_tree(data0, labels0, criterion, min_split), build_tree(data1, labels1, criterion, min_split))

def apply_tree(tree, features):
    '''
    conf = apply_tree(tree, features)

    Applies the decision tree to a set of features.
    '''
    if type(tree) is _Leaf:
        return tree.v
    i,c,left,right = tree
    if features[i] < c:
        return apply_tree(left, features)
    return apply_tree(right, features)


class tree_classifier(object):
    '''
    tree = tree_classifier()
    tree.train(features,labels)
    predicted = tree.apply(testfeatures)

    A decision tree classifier (currently, implements the greedy ID3 
    algorithm without any pruning.

    Attributes
    ----------
        * criterion: criterion to use for tree construction,
            this should be a function that receives a set of labels
            (default: information_gain).
        * min_split: minimum size to split on (default: 4).
    '''
    def __init__(self, criterion=information_gain, min_split=4, return_label=True):
        self.criterion = criterion
        self.min_split = 4
        self.return_label = return_label

    def train(self, features, labels):
        return tree_model(build_tree(features, labels, self.criterion, self.min_split), self.return_label)

class tree_model(object):
    def __init__(self, tree, return_label):
        self.tree = tree
        self.return_label = return_label

    def apply(self,feats):
        value = apply_tree(self.tree, feats)
        if self.return_label:
            return value > .5
        return value

