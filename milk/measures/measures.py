# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# License: MIT

from __future__ import division
import numpy
import numpy as np

__all__ = ['accuracy','waccuracy']

def accuracy(cmatrix):
    '''
    acc = accuracy(cmatrix)

    Accuracy of confusion matrix
    '''
    cmatrix = numpy.asanyarray(cmatrix)
    return cmatrix.trace()/cmatrix.sum()

def zero_one_loss(real, predicted):
    '''
    loss = zero_one_loss(real, predicted)

    Parameters
    ----------
    real : sequence
        the underlying labels
    predicted : sequence
        the predicted labels

    Returns
    -------
    loss : integer
        the number of instances where `real` differs from `predicted`
    '''
    return np.sum(np.asanyarray(real) != np.asanyarray(predicted))
    

def waccuracy(cmatrix):
    '''
    wacc = waccuracy(cmatrix)

    Weighted accuracy of cmatrix
    '''
    cmatrix = numpy.asanyarray(cmatrix)
    return (cmatrix.diagonal() / cmatrix.sum(1)).mean()

## TODO: Implement http://en.wikipedia.org/wiki/Matthews_Correlation_Coefficient


