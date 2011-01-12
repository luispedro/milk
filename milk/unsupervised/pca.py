# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from numpy import linalg
from . import normalise

__all__ = [
    'pca',
    ]
    
def pca(X, zscore=True):
    '''
    Y,V = pca(X, zscore=True)

    Principal Component Analysis

    Performs principal component analysis. Returns transformed
    matrix and principal components

    Parameters
    ----------
    X : 2-dimensional ndarray
        data matrix
    zscore : boolean, optional
        whether to normalise to zscores (default: True)

    Returns
    -------
    Y : ndarray
        Transformed matrix (of same dimension as X)
    V : ndarray
        principal components
    '''
    if zscore:
        X = normalise.zscore(X)
    C = np.cov(X.T)
    w,v = linalg.eig(C)
    Y = np.dot(v,X.T).T
    return Y,v

