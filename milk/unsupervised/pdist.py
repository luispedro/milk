# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np

__all__ = [
    'pdist',
    'plike',
    ]

def pdist(X, Y=None, distance='euclidean2'):
    '''
    D = pdist(X, Y={X}, distance='euclidean2')

    Compute distance matrix::

    D[i,j] == np.sum( (X[i] - Y[j])**2 )

    Parameters
    ----------
      X : feature matrix
      Y : feature matrix (default: use `X`)
      distance : one of 'euclidean' or 'euclidean2' (default)

    Returns
    -------
      D : matrix of doubles
    '''
    # Use Dij = np.dot(Xi, Xi) + np.dot(Xj,Xj) - 2.*np.dot(Xi,Xj)
    if Y is None:
        D = np.dot(X, X.T)
        x2 = D.diagonal()
        y2 = x2
    else:
        D = np.dot(X, Y.T)
        x2 = np.array([np.dot(x,x) for x in X])
        y2 = np.array([np.dot(y,y) for y in Y])
    D *= -2.
    D += x2[:,np.newaxis]
    D += y2

    # Because of numerical imprecision, we might get negative numbers
    # (which cause problems down the road, e.g., when doing the sqrt):
    np.maximum(D, 0, D)
    if distance == 'euclidean':
        np.sqrt(D, D)
    return D


def plike(X, sigma2=None):
    '''
    L = plike(X, sigma2={guess based on X})

    Parameters
    ----------
      X : feature matrix
      sigma2 : bandwidth

    Returns
    -------
      L : likelihood matrix
    '''

    L = pdist(X)
    if sigma2 is None:
        sigma2 = np.median(L)
    L /= -sigma2
    np.exp(L, L)
    return L
