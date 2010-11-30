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
from ..supervised.classifier import normaliselabels
import numpy as np

__all__ = ['foldgenerator', 'getfold', 'nfoldcrossvalidation']
def foldgenerator(labels, nfolds=None, origins=None):
    '''
    for train,test in foldgenerator(labels, nfolds=None, origins=None)
        ...

    This generator breaks up the data into `n` folds (default 10).

    If `origins` is given, then all elements that share the same origin will
    either be in testing or in training (never in both).  This is useful when
    you have several replicates that shouldn't be mixed together between
    training&testing but that can be otherwise be treated as independent for
    learning.

    Parameters
    ----------
    labels : a sequence
        the labels
    nfolds : integer
        nr of folds (default 10 or minimum label size)
    origins : if present, must be an array of indices of the same size as labels.

    Returns
    -------
    iterator over `train, test`, two boolean arrays

    Bugs
    ----
    Algorithm is very naive w.r.t. unbalanced origins.
    '''
    labels,_ = normaliselabels(labels)
    nlabels = labels.max()+1
    label_locations = [np.where(labels == i)[0] for i in xrange(nlabels)]
    label_counts = np.array(map(len, label_locations))
    if origins is not None:
        origins = np.asanyarray(origins)
        if len(origins) != len(labels):
            raise ValueError(
             'milk.nfoldcrossvalidation.foldgenerator: origins must be of same size as labels')
        label_counts_orig = np.zeros(nlabels, int)
        label_origins = [[] for i in xrange(nlabels)]
        counted = set()
        for lab, orig in zip(labels, origins):
            if orig in counted:
                if orig not in label_origins[lab]:
                    for other in xrange(nlabels):
                        if other == lab: continue
                        if orig in label_origins[other]:
                            raise ValueError('milk.nfoldcrossvalidation.foldgenerator: origin %s seems to be present in two labels %s and %s' % (orig, lab, other))
                    raise AssertionError('This should have been unreachable code')
                continue
            label_counts_orig[lab] += 1
            label_origins[lab].append(orig)
            counted.add(orig)
        label_origins_counts = [np.cumsum([(origins == orig).sum() for orig in label_origins[lab]])
                        for lab in xrange(nlabels)]
    else:
        label_counts_orig = label_counts

    if nfolds is None:
        nfolds = min(label_counts_orig.min(), 10)
    elif nfolds > label_counts_orig.min():
        from warnings import warn
        warn('milk.measures.nfoldcrossvalidation: Reducing the nr. of folds from %s to %s (minimum class size).' % (nfolds, label_counts_orig.min()))
        nfolds = label_counts_orig.min()
    if nfolds == 1:
        note = ('(taking `origins` into account)' if origins is not None else '')
        raise ValueError('''
milk.nfoldcrossvalidation.foldgenerator: nfolds was reduced to 1 because minimum class size was 1.

If you passed in an origins parameter, it might be caused by having a class come from a single origin.

The class histogram %s looks like:
%s''' % (note, label_counts_orig))

    perfolds = (label_counts // nfolds)
    assert perfolds.min() > 0
    testing = np.zeros(len(labels), bool)
    start = np.zeros(nlabels, int)
    for fold in xrange(nfolds):
        testing.fill(False)
        end = start + perfolds
        # We need to force the last fold to be past the end to avoid rounding errors
        # missing some elements
        if fold == (nfolds - 1):
            end.fill(len(labels))
        for i,locs,s,e in zip(xrange(nlabels), label_locations, start, end):
            if origins is None:
                testing[locs[s:e]] = 1
            else:
                s,e = np.searchsorted(label_origins_counts[i], (s,e))
                for included in label_origins[i][s:e]:
                    testing[origins == included] = True
        yield ~testing, testing
        start = end


def getfold(labels, fold, nfolds=None, origins=None):
    '''
    trainingset,testingset = getfold(labels, fold, nfolds=None, origins=None)

    Get the training and testing set for fold `fold` in `nfolds`

    Arguments are the same as for `foldgenerator`

    Parameters
    ----------
    labels : ndarray of labels
    fold : integer
    nfolds : integer
        number of folds (default 10 or size of smallest class)
    origins : sequence, optional
        if given, then objects with same origin are *not* scattered across folds
    '''
    if nfolds < fold:
        raise ValueError('milk.getfold: Attempted to get fold %s out of %s' % (fold, nfolds))
    for i,(t,s) in enumerate(foldgenerator(labels, nfolds, origins)):
        if i == fold:
            return t,s
    raise ValueError('milk.getfold: Attempted to get fold %s but the number of actual folds was too small (%s)' % (fold,i))

def nfoldcrossvalidation(features, labels, nfolds=None, classifier=None, origins=None, return_predictions=False):
    '''
    Perform n-fold cross validation

    cmatrix,names = nfoldcrossvalidation(features, labels, nfolds=10, classifier={defaultclassifier()}, origins=None, return_predictions=False)
    cmatrix,names,predictions = nfoldcrossvalidation(features, labels, nfolds=10, classifier={defaultclassifier()}, origins=None, return_predictions=True)

    cmatrix will be a N x N matrix, where N is the number of classes

    cmatrix[i,j] will be the number of times that an element of class i was
    classified as class j

    names[i] will correspond to the label name of class i

    Parameters
    ----------
    features : a sequence
    labels : an array of labels, where label[i] is the label corresponding to features[i]
    nfolds : integer, optional
        Nr of folds. Default: 10
    classifier : learner object, optional
        classifier should implement the train() method to return a model
        (something with an apply() method). defaultclassifier() by default

    origins : sequence, optional
        Origin ID (see foldgenerator)
    return_predictions : bool, optional
        whether to return predictions (default: False)

    Returns
    -------
    cmatrix : ndarray
        confusion matrix
    names : sequence
        sequence of labels so that cmatrix[i,j] corresponds to names[i], names[j]
    predictions : sequence
        predicted output for each element
    '''
    import operator
    from .measures import confusion_matrix
    if len(features) != len(labels):
        raise ValueError('milk.measures.nfoldcrossvalidation: len(features) should match len(labels)')
    if classifier is None:
        from ..supervised.defaultclassifier import defaultclassifier
        classifier = defaultclassifier()
    labels,names = normaliselabels(labels)
    if return_predictions:
        predictions = np.empty_like(labels)
        predictions.fill(-1) # This makes it clearer if there are bugs in the programme

    features = np.asanyarray(features)

    nclasses = labels.max() + 1
    results = []
    measure = confusion_matrix
    for trainingset,testingset in foldgenerator(labels, nfolds, origins=origins):
        model = classifier.train(features[trainingset], labels[trainingset])
        cur_predictions = [model.apply(f) for f in features[testingset]]
        if return_predictions:
            predictions[testingset] = np.array(cur_predictions, dtype=predictions.dtype)
        results.append(measure(labels[testingset], cur_predictions))

    result = reduce(operator.add, results)
    if return_predictions:
        return result, names, predictions
    return result, names

