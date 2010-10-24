# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010, Luis Pedro Coelho <lpc@cmu.edu>
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
from collections import defaultdict
import numpy as np


class kNN(object):
    '''
    k-Nearest Neighbour Classifier

    Naive implementation of a k-nearest neighbour classifier.

    C = kNN(k)

    Attributes:
    -----------
        * k: the k to use
    '''


    def __init__(self, k=1):
        self.k = k

    def train(self, features, labels, normalisedlabels=False, copy_features=False):
        features = np.asanyarray(features)
        labels = np.asanyarray(labels)
        if copy_features:
            features = features.copy()
            labels = labels.copy()
        return kNN_model(self.k, features, labels)

class kNN_model(object):
    def __init__(self, k, features, labels):
        self.k = k
        self.features = features
        self.labels = labels

    def apply(self,features):
        diff2 = ( (self.features - features)**2 ).sum(1)
        neighbours = diff2.argsort()[:self.k]
        labels = self.labels[neighbours]
        votes = defaultdict(int)
        for L in labels:
            votes[L] += 1
        v,L = max( (v,L) for L,v in votes.items() )
        return L


# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
