# -*- coding: utf-8 -*-
# Copyright (C) 2008-2012, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from numpy import linalg
from . import normalise

__all__ = [
    'pca',
    'mds',
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


def mds(features, ndims, zscore=False):
    '''
    X = mds(features, ndims, zscore=False)

    Euclidean Multi-dimensional Scaling

    Parameters
    ----------
    features : ndarray
        data matrix
    ndims : int
        Number of dimensions to return
    zscore : boolean, optional
        Whether to zscore the features (default: False)

    Returns
    -------
    X : ndarray
        array of size ``(m, ndims)`` where ``m = len(features)``
    '''
    if zscore:
        X = normalise.zscore(features)
    dists = milk.unsupervised.pdist(features)
    mu = dists.mean()
    dists -= dists.mean(0)
    dists = dists.T
    dists -= dists.mean(0)
    dists = dists.T
    dists += mu

    u,s,v = np.linalg.svd(dists)
    s = np.diag(np.sqrt(s))
    X = np.dot(u, s)
    return X[:,1:(ndims+1)]

