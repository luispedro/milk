# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# License: MIT

from __future__ import division
import numpy as np

__all__ = [
    'accuracy',
    'confusion_matrix',
    'waccuracy',
    'zero_one_loss',
    ]

def accuracy(real, other=None, normalisedlabels=False, names=None):
    '''
    acc = accuracy(real, predicted, normalisedlabels=False, names=None)

    Compute accuracy (fraction of correct predictions).

    Parameters
    ----------
    real : sequence
        The real labels
    predicted : sequence
        The predicted sequence (must be same type as `real`)
    normalisedlabels : boolean, optional
        Whether labels have been normalised
    names : sequence
        The underlying names (unused)

    Returns
    -------
    acc : float
    '''
    if other is None:
        import warnings
        warnings.warn('milk.measures.accuracy: calling this with one argument is a deprecated interface.', DeprecationWarning)
        cmatrix = np.asanyarray(real)
        return cmatrix.trace()/cmatrix.sum()
    else:
        return np.mean(np.asanyarray(real) == other)

def zero_one_loss(real, predicted, normalisedlabels=False, names=None):
    '''
    loss = zero_one_loss(real, predicted, normalisedlabels={unused}, names={unused})

    Parameters
    ----------
    real : sequence
        the underlying labels
    predicted : sequence
        the predicted labels
    normalisedlabels : unused
    names: unused

    Returns
    -------
    loss : integer
        the number of instances where `real` differs from `predicted`
    '''
    return np.sum(np.asanyarray(real) != np.asanyarray(predicted))


def waccuracy(real, predicted=None, normalisedlabels=False, names=None):
    '''
    wacc = waccuracy(real, predicted, normalisedlabels={unused}, names={unused})

    Weighted accuracy: average of accuracy for each (real) class. Can be very
    different from accuracy if the classes are unbalanced (in particular, if
    they are very unbalanced, you can get a high accuracy with a bad
    classifier).

    Parameters
    ----------
    real : sequence
        the underlying labels
    predicted : sequence
        the predicted labels
    normalisedlabels : unused
    names: unused

    Returns
    -------
    wacc : float
        the weighted accuracy
    '''
    if predicted is None:
        import warnings
        warnings.warn('milk.measures.accuracy: calling this with one argument is a deprecated interface.', DeprecationWarning)
        cmatrix = np.asanyarray(real)
    else:
        cmatrix = confusion_matrix(real, predicted, normalisedlabels, names)
    return (cmatrix.diagonal() / cmatrix.sum(1)).mean()

def confusion_matrix(real, predicted, normalisedlabels=False, names=None):
    '''
    cmatrix = confusion_matrix(real, predicted, normalisedlabels=False, names=None)

    Computes the confusion matrix

    Parameters
    ----------
    real : sequence
        The real labels
    predicted : sequence
        The predicted sequence (must be same type as `real`)
    normalisedlabels : boolean, optional
        Whether labels have been normalised
    names : sequence
        The underlying names (unused)

    Returns
    -------
    cmatrix : 2 ndarray
    '''
    if not normalisedlabels:
        from ..supervised.normalise import normaliselabels
        real, names = normaliselabels(real)
        predicted = map(names.index, predicted)
    n = np.max(real)+1
    cmat = np.zeros((n,n), int)
    for r,p in zip(real, predicted):
        cmat[r,p] += 1
    return cmat


def bayesian_significance(n, c0, c1):
    '''
    sig = bayesian_significance(n, c0, c1)

    Computes the Bayesian significance of the difference between a classifier
    that gets ``c0`` correct versus one that gets ``c1`` right on ``n``
    examples.

    Parameters
    ----------
    n : int
        Total number of examples
    c0 : int
        Examples that first classifier got correct
    c1 : int
        Examples that second classifier got correct

    Returns
    -------
    sig : float
        Significance value
    '''
    def _logp(r, n, c):
        return c*np.log(r)+(n-c)*np.log(1-r)
    r = np.linspace(.0001,.9999,100)
    lp0 = _logp(r,n,c0)
    lp1 = _logp(r,n,c1)
    mat = lp0 + lp1[:,np.newaxis]
    mat -= mat.max()
    mat = np.exp(mat)
    mat /= mat.sum()
    sig = np.triu(mat).sum()-mat.trace()/2.
    return min(sig, 1. - sig)


## TODO: Implement http://en.wikipedia.org/wiki/Matthews_Correlation_Coefficient
