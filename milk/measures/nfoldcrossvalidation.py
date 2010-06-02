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
from collections import defaultdict
from ..supervised.classifier import normaliselabels
from ..supervised.defaultclassifier import defaultclassifier
import numpy
import numpy as np

__all__ = ['foldgenerator', 'getfold', 'nfoldcrossvalidation']
def foldgenerator(labels, nfolds=None, origins=None, is_ordered=False):
    '''
    for train,test in foldgenerator(labels, nfolds=None, origins=None, is_ordered=False)
        ...

    This generator breaks up the data into `n` folds (default 10).

    If `origins` is given, then all elements that share the same origin will
    either be in testing or in training (never in both).  This is useful when
    you have several replicates that shouldn't be mixed together between
    training&testing but that can be otherwise be treated as independent for
    learning.

    Parameters
    ----------
      labels : the labels
      nfolds : nr of folds (default 10)
      origins : if present, must be an array of indices of the same size as labels.
      is_ordered : whether input is in canonical format: i.e., labels/origins
                   are 0..N-1/0..K-1, in order. If True, this can be faster
    Returns
    -------
      iterator over `train, test`, two boolean arrays
    Bugs
    ----
    Algorithm is very naive w.r.t. unbalanced origins.

    This is  slower than it could be if `origins is None`.
    '''

    if origins is None:
        origins = np.arange(len(labels))
    assert len(origins) == len(labels), \
         'milk.nfoldcrossvalidation.foldgenerator: origins must be of same size as labels'
    counted = set()
    classcounts = defaultdict(int)
    for L,orig in zip(labels,origins):
        if orig in counted: continue
        classcounts[L] += 1
        counted.add(orig)
    min_class_count = min(classcounts.values())
    if nfolds is None:
        nfolds = min(10, min_class_count)
    elif min_class_count < nfolds:
        from warnings import warn
        warn('milk.measures.nfoldcrossvalidation: Reducing the nr. of folds to %s (minimum class size).' % min_class_count)
        nfolds = min_class_count

    if not is_ordered:
        labels,_ = normaliselabels(labels)
        reorder = None
        if np.any(np.diff(labels) < 0):
            reorder = range(len(labels))
            reorder.sort(key=(lambda i: (labels[i], origins[i])))
            reorder = np.array(reorder)
            labels = labels[reorder]
            origins = origins[reorder]
            for i,ri in enumerate(reorder.copy()):
                reorder[ri] = i
        origins_ordered = np.empty_like(origins)
        index = defaultdict(xrange(len(origins)).__iter__().next)
        for idx,orig in enumerate(origins):
            origins_ordered[idx] = index[orig]
        origins = origins_ordered
        if reorder is not None:
            for Tr,Te in foldgenerator(labels, nfolds, origins, is_ordered=True):
                yield Tr[reorder], Te[reorder]
            return

    nlabels = labels.max() + 1
    weights = np.zeros(origins.max()+1, np.float)
    label_origin = np.zeros(origins.max()+1, np.float)
    origin_labels = [np.where(labels == L)[0] for L in xrange(nlabels)]
    offset = [origins[labels == L].min() for L in xrange(nlabels)]
    perfold = 1./nfolds
    for lab,orig in zip(labels, origins):
        label_origin[orig] = lab
        weights[orig] += 1

    for lab in set(labels):
        is_lab = (label_origin == lab)
        weights[is_lab] /= (weights * is_lab).sum()
        weights[is_lab] = np.cumsum(weights[is_lab])

    testing = np.empty(len(labels), np.bool)
    for fold in xrange(nfolds):
        testing.fill(False)
        for label in xrange(nlabels):
            start = perfold * fold
            # We need to force the last fold to be way past the end to avoid rounding errors
            # missing some elements
            end = perfold * (fold+1) + (fold == nfolds-1)
            start,end = np.searchsorted(weights[label_origin == label], (start, end))
            start += offset[label]
            end += offset[label]
            testing |= (origins >= start)&(origins<end)
        yield ~testing, testing

def getfold(labels, fold, nfolds=None, origins=None, is_ordered=False):
    '''
    trainingset,testingset = getfold(labels, fold, nfolds=None, origins=None, is_ordered=False)

    Get the training and testing set for fold `fold` in `nfolds`

    Arguments are the same as for `foldgenerator`
    '''
    nfolds = (10 if nfolds is None else nfolds)
    assert fold < nfolds, 'milk.getfold: Attempted to get fold %s out of %s' % (fold, nfolds)
    for i,(t,s) in enumerate(foldgenerator(labels, nfolds, origins, is_ordered)):
        if i == fold:
            return t,s
    assert False, 'milk.getfold: Attempted to get fold %s but the number of actual folds was too small' % fold

def nfoldcrossvalidation(features, labels, nfolds=None, classifier=None, origins=None, return_predictions=False):
    '''
    Perform n-fold cross validation

    cmatrix,labelnames = nfoldcrossvalidation(features, labels, nfolds=10, classifier={defaultclassifier()}, origins=None, return_predictions=False)
    cmatrix,labelnames,predictions = nfoldcrossvalidation(features, labels, nfolds=10, classifier={defaultclassifier()}, origins=None, return_predictions=False)

    cmatrix will be a N x N matrix, where N is the number of classes
    cmatrix[i,j] will be the number of times that an element of class i was classified as class j

    labelnames[i] will correspond to the label name of class i

    Parameters
    ----------
      features : a feature matrix or list of feature vectors
      labels : an array of labels, where label[i] is the label corresponding to features[i]
      nfolds : Nr of folds.
      origins : Origin ID (see foldgenerator)
      
      return_predictions : whether to return predictions

    classifier should implement the train() method to return a model (something with an apply() method)
    '''
    assert len(features) == len(labels), 'milk.measures.nfoldcrossvalidation: len(features) should match len(labels)'
    if classifier is None:
        classifier = defaultclassifier()
    labels,labelnames = normaliselabels(labels)
    predictions = np.zeros_like(labels)-1

    features = numpy.asanyarray(features)

    nclasses = labels.max() + 1
    cmatrix = np.zeros((nclasses,nclasses))
    for trainingset,testingset in foldgenerator(labels, nfolds, origins=origins):
        model = classifier.train(features[trainingset], labels[trainingset])
        prediction = np.array([model.apply(f) for f in features[testingset]])
        predictions[testingset] = prediction.astype(predictions.dtype)
        for p, r in zip(prediction,labels[testingset]):
            cmatrix[r,p] += 1

    if return_predictions:
        return cmatrix, labelnames, predictions
    return cmatrix, labelnames

