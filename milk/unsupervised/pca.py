# -*- coding: utf-8 -*-
# Copyright (C) 2008-2013, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from numpy import linalg
from . import normalise
from .pdist import pdist

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

    See Also
    --------
    mds_dists : function
    '''
    if zscore:
        features = normalise.zscore(features)
    else:
        features = np.asarray(features)
    P2 = pdist(features)
    return mds_dists(P2, ndims)

def mds_dists(distances, ndims):
    '''
    X = mds_dists(distances, ndims)

    Euclidean Multi-dimensional Scaling based on a distance matrix

    Parameters
    ----------
    distances : ndarray
        data matrix
    ndims : int
        Number of dimensions to return

    Returns
    -------
    X : ndarray
        array of size ``(m, ndims)`` where ``m = len(features)``

    See Also
    --------
    mds : function
    '''

    n = len(distances)
    J = np.eye(n) - (1./n)* np.ones((n,n))
    B = -.5 * np.dot(J,np.dot(distances,J))
    w,v = np.linalg.eig(B)


    w = w[:ndims]
    s = np.sign(w)
    w = np.abs(w).real
    w = np.diag(np.sqrt(s * w))
    X = np.dot(v[:,:ndims], w)
    return X.real

