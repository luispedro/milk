# -*- coding: utf-8 -*-
# Copyright (C) 2008-2012, Luis Pedro Coelho <luis@luispedro.org>
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from numpy.linalg import det
from . classifier import normaliselabels

__all__ = [
    'sda',
    'linearly_independent_subset',
    'linear_independent_features',
    'filterfeatures',
    'featureselector',
    'sda_filter',
    ]

_TOLERANCE = 0
_SIGNIFICANCE_IN = .15
_SIGNIFICANCE_OUT = .15

def _sweep(A, k, flag):
    Akk = A[k,k]
    if Akk == 0:
        Akk = 1.e-5

    # cross[i,j] = A[i,k] * A[k,j]
    cross = (A[:,k][:, np.newaxis] * A[k])
    B = A - cross/Akk

    # currently: B[i,j] = A[i,j] - A[i,k]*A[k,j]/Akk
    # Now fix row k and col k, followed by Bkk
    B[k] = flag * A[k]/A[k,k]
    B[:,k] = flag * A[:,k]/A[k,k]
    B[k,k] = -1./Akk
    return B

def sda(features, labels, tolerance=None, significance_in=None, significance_out=None):
    '''
    features_idx = sda(features,labels)

    Perform Stepwise Discriminant Analysis for feature selection

    Pre filter the feature matrix to remove linearly dependent features
    before calling this function. Behaviour is undefined otherwise.

    This implements the algorithm described in
    Jennrich, R.I. (1977), "Stepwise Regression" & "Stepwise Discriminant Analysis,"
    both in Statistical Methods for Digital Computers, eds.
    K. Enslein, A. Ralston, and H. Wilf, New York; John Wiley & Sons, Inc.
    '''
    import scipy.stats

    if tolerance is None:
        tolerance = _TOLERANCE
    if significance_in is None:
        significance_in = _SIGNIFICANCE_IN
    if significance_out is None:
        significance_out = _SIGNIFICANCE_OUT

    assert len(features) == len(labels), 'milk.supervised.featureselection.sda: length of features not the same as length of labels'
    N, m = features.shape
    labels,labelsu = normaliselabels(labels)
    q = len(labelsu)

    df = features - features.mean(0)
    T = np.dot(df.T, df)

    dfs = [(features[labels == i] - features[labels == i].mean(0)) for i in xrange(q)]
    W = np.sum(np.dot(d.T, d) for d in dfs)

    ignoreidx = ( W.diagonal() == 0 )
    if ignoreidx.any():
        idxs, = np.where(~ignoreidx)
        F = sda(features[:,~ignoreidx],labels)
        if len(F): return idxs[F]
        return ignoreidx
    output = []
    D = W.diagonal()
    df1 = q-1
    last_enter_k = -1
    while True:
        V = W.diagonal()/T.diagonal()
        W_d = W.diagonal()
        V_neg = (W_d < 0)
        p = V_neg.sum()
        if V_neg.any():
            V_m = V[V_neg].min()
            k, = np.where(V == V_m)
            k = k[0]
            Fremove = (N-p-q+1)/(q-1)*(V_m-1)
            df2 = N-p-q+1
            PrF = 1 - scipy.stats.f.cdf(Fremove,df1,df2)
            if PrF > significance_out:
                #print 'removing ',k, 'V(k)', 1./V_m, 'Fremove', Fremove, 'df1', df1, 'df2', df2, 'PrF', PrF
                if k == last_enter_k:
                    # We are going into an infinite loop.
                    import warnings
                    warnings.warn('milk.featureselection.sda: infinite loop detected (maybe bug?).')
                    break
                W = _sweep(W,k,1)
                T = _sweep(T,k,1)
                continue
        ks = ( (W_d / D) > tolerance)
        if ks.any():
            V_m = V[ks].min()
            k, = np.where(V==V_m)
            k = k[0]
            Fenter = (N-p-q)/(q-1) * (1-V_m)/V_m
            df2 = N-p-q
            PrF = 1 - scipy.stats.f.cdf(Fenter,df1,df2)
            if PrF < significance_in:
                #print 'adding ',k, 'V(k)', 1./V_m, 'Fenter', Fenter, 'df1', df1, 'df2', df2, 'PrF', PrF
                W = _sweep(W,k,-1)
                T = _sweep(T,k,-1)
                if PrF < .0001:
                    output.append((Fenter,k))
                last_enter_k = k
                continue
        break

    output.sort(reverse=True)
    return np.array([x[1] for x in output])


def linearly_independent_subset(V, threshold=1.e-5, return_orthogonal_basis=False):
    '''
    subset = linearly_independent_subset(V, threshold=1.e-5)
    subset,U = linearly_independent_subset(V, threshold=1.e-5, return_orthogonal_basis=True)

    Discover a linearly independent subset of `V`

    Parameters
    ----------
    V : sequence of input vectors
    threshold : float, optional
        vectors with 2-norm smaller or equal to this are considered zero
        (default: 1e.-5)
    return_orthogonal_basis : Boolean, optional
        whether to return orthogonal basis set

    Returns
    -------
    subset : ndarray of integers
        indices used for basis
    U : 2-array
        orthogonal basis into span{V}

    Implementation Reference
    ------------------------
    Use Gram-Schmidt with a check for when the v_k is close enough to zero to ignore

    See http://en.wikipedia.org/wiki/Gram-Schmidt_process
    '''
    V = np.array(V, copy=True)
    orthogonal = []
    used = []
    for i,u in enumerate(V):
        for v in orthogonal:
            u -= np.dot(u,v)/np.dot(v,v) * v
        if np.dot(u,u) > threshold:
            orthogonal.append(u)
            used.append(i)
    if return_orthogonal_basis:
        return np.array(used),np.array(orthogonal)
    return np.array(used)


def linear_independent_features(features, labels=None):
    '''
    indices = linear_independent_features(features, labels=None)

    Returns the indices of a set of linearly independent features (columns).

    Parameters
    ----------
    features : ndarray
    labels : ignored
        This argument is only here to conform to the learner interface.

    Returns
    -------
    indices : ndarray of integers
        indices of features to keep

    See Also
    --------
    `linearly_independent_subset` :
        this function is equivalent to `linearly_independent_subset(features.T)`
    '''
    return linearly_independent_subset(features.T)


class filterfeatures(object):
    '''
    selector = filterfeatures(idxs)

    Returns a transformer which selects the features given
     by idxs
    '''
    def __init__(self, idxs):
        self.idxs = idxs

    def apply(self, features):
        return features[self.idxs]

    def __repr__(self):
        return 'filterfeatures(%s)' % self.idxs

class featureselector(object):
    '''
    selector = featureselector(function)

    Returns a transformer which selects features according to
        selected_idxs = function(features,labels)
    '''
    def __init__(self, selector):
        self.selector = selector

    def train(self, features, labels, **kwargs):
        idxs = self.selector(features, labels)
        if len(idxs) == 0:
            import warnings
            warnings.warn('milk.featureselection: No features selected! Using all features as fall-back.')
            idxs = np.arange(len(features[0]))
        return filterfeatures(idxs)

    def __repr__(self):
        return 'featureselector(%s)' % self.selector

def sda_filter():
    return featureselector(sda)

