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
from collections import defaultdict
from ..supervised.classifier import normaliselabels
import numpy
import numpy as np

__all__=['nfoldcrossvalidation']
def foldgenerator(labels,nfolds=None):
    '''
    for test,train in foldgenerator(labels,nfolds=None)
        ...

    This generator breaks up the data into nfolds (default 10).

    Params
    ------
        * labels: the labels
        * nfolds: nr of folds (default 10)
    '''
    classcounts = defaultdict(int)
    for L in labels:
        classcounts[L] += 1

    min_class_count = min(classcounts.values())
    if nfolds is None:
        nfolds = min(10,min_class_count)
    elif min_class_count < nfolds:
        from warnings import warn
        warn('milk.measures.nfoldcrossvalidation: Reducing the nr. of folds to %s (minimum class size).' % min_class_count)
        nfolds = min_class_count
    
    testingset = np.empty(len(labels),bool)
    for fold in xrange(nfolds):
        testingset.fill(False)
        for L,C in classcounts.items():
            idxs, = np.where(labels==L)
            N = len(idxs)
            perfold = N/nfolds
            start = np.floor(perfold*fold)
            end = np.floor(perfold*(fold+1))
            idxs = idxs[start:end]
            testingset[idxs]=True
        trainingset = ~testingset
        yield trainingset,testingset

def nfoldcrossvalidation(features,labels,nfolds=None,classifier=None, return_predictions=False):
    '''
    Perform n-fold cross validation

    cmatrix,labelnames = nfoldcrossvalidation(features, labels, nfolds=10, classifier=None, return_predictions=False)
    cmatrix,labelnames,predictions = nfoldcrossvalidation(features, labels, nfolds=10, classifier=None, return_predictions=False)

    cmatrix will be a N x N matrix, where N is the number of classes
    cmatrix[i,j] will be the number of times that an element of class i was classified as class j

    labelnames[i] will correspond to the label name of class i

    Arguments
    ---------
        * features: a feature matrix or list of feature vectors
        * labels: an array of labels, where label[i] is the label corresponding to features[i]
        * nfolds: Nr of folds.
        * return_predictions: whether to return predictions

    classifier should implement the train() and apply() methods
    '''
    assert len(features) == len(labels), 'milk.measures.nfoldcrossvalidation: len(features) should match len(labels)'
    if classifier is None:
        classifier = defaultclassifier()
    labels,labelnames = normaliselabels(labels)
    predictions = np.zeros_like(labels)-1

    features = numpy.asanyarray(features)

    nclasses = labels.max() + 1
    cmatrix = np.zeros((nclasses,nclasses))
    for trainingset,testingset in foldgenerator(labels, nfolds):
        classifier.train(features[trainingset], labels[trainingset])
        prediction = np.array([classifier.apply(f) for f in features[testingset]])
        predictions[testingset] = prediction
        for p, r in zip(prediction,labels[testingset]):
            cmatrix[r,p] += 1

    if return_predictions:
        return cmatrix, labelnames, predictions
    return cmatrix, labelnames

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
