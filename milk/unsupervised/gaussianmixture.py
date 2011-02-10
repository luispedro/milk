# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from numpy import log, pi, array
from numpy.linalg import det, inv
from kmeans import residual_sum_squares, centroid_errors

__all__ = [
    'BIC',
    'AIC',
    'log_likelihood',
    'nr_parameters',
    ]

def log_likelihood(fmatrix,assignments,centroids,model='one_variance',covs=None):
    '''
    log_like = log_likelihood(feature_matrix, assignments, centroids, model='one_variance', covs=None)

    Compute the log likelihood of feature_matrix[i] being generated from centroid[i]
    '''
    N,q = fmatrix.shape
    k = len(centroids)
    if model == 'one_variance':
        Rss = residual_sum_squares(fmatrix,assignments,centroids)
        #sigma2=Rss/N
        return -N/2.*log(2*pi*Rss/N)-N/2
    elif model == 'diagonal_covariance':
        errors = centroid_errors(fmatrix,assignments,centroids)
        errors *= errors
        errors = errors.sum(1)
        Rss = np.zeros(k)
        counts = np.zeros(k)
        for i in xrange(fmatrix.shape[0]):
            c = assignments[i]
            Rss[c] += errors[i]
            counts[c] += 1
        sigma2s = Rss/(counts+(counts==0))
        return -N/2.*log(2*pi) -N/2. -1/2.*np.sum(counts*np.log(sigma2s+(counts==0)))
    elif model == 'full_covariance':
        res = -N*q/2.*log(2*pi)

        for k in xrange(len(centroids)):
            diff = (fmatrix[assignments == k] - centroids[k])
            if covs is None:
                covm = np.cov(diff.T)
            else:
                covm = covs[k]
            if covm.shape == ():
                covm = np.matrix([[covm]])
            icov = np.matrix(inv(covm))
            diff = np.matrix(diff)
            Nk = diff.shape[0]
            res += -Nk/2.*log(det(covm)) + \
                 -.5 * (diff * icov * diff.T).diagonal().sum() 
        return res

    raise ValueError, "log_likelihood: cannot handle model '%s'" % model


def nr_parameters(fmatrix,k,model='one_variance'):
    '''
    nr_p = nr_parameters(fmatrix, k, model='one_variance')

    Compute the number of parameters for a model of k clusters on

    Parameters
    ----------
    fmatrix : 2d-array
        feature matrix
    k : integer
        nr of clusters
    model : str
        one of 'one_variance' (default), 'diagonal_covariance', or 'full_covariance'

    Returns
    -------
    nr_p : integer
        Number of parameters
    '''
    N,q = fmatrix.shape
    if model == 'one_variance':
        return k*q+1
    elif model == 'diagonal_covariance':
        return k*(q+1)
    elif model == 'full_covariance':
        return k*+q*q

    raise ValueError, "milk.unsupervised.gaussianmixture.nr_parameters: cannot handle model '%s'" % model

def _compute(type, fmatrix, assignments, centroids, model='one_variance', covs=None):
    N,q = fmatrix.shape
    k = len(centroids)
    log_like = log_likelihood(fmatrix, assignments, centroids, model, covs)
    n_param = nr_parameters(fmatrix,k,model)
    if type == 'BIC':
        return -2*log_like + n_param * log(N)
    elif type == 'AIC':
        return -2*log_like + 2 * n_param
    else:
        assert False

def BIC(fmatrix, assignments, centroids, model='one_variance', covs=None):
    '''
    B = BIC(fmatrix, assignments, centroids, model='one_variance', covs={From Data})

    Compute Bayesian Information Criterion

    Parameters
    ----------
    fmatrix : 2d-array
        feature matrix
    assignments : 2d-array
        Centroid assignments
    centroids : sequence
        Centroids
    model : str, optional
        one of

        'one_variance'
            All features share the same variance parameter sigma^2. Default

        'full_covariance'
            Estimate a full covariance matrix or use covs[i] for centroid[i]
    covs : sequence or matrix, optional
        Covariance matrices. If None, then estimate from the data. If scalars
        instead of matrices are given, then s stands for sI (i.e., the diagonal
        matrix with s along the diagonal).

    Returns
    -------
    B : float
        BIC value

    See Also
    --------
    AIC
    '''
    return _compute('BIC', fmatrix, assignments, centroids, model, covs)

def AIC(fmatrix,assignments,centroids,model='one_variance',covs=None):
    '''
    A = AIC(fmatrix,assignments,centroids,model)

    Compute Akaike Information Criterion

    Parameters
    ----------
    fmatrix : 2d-array
        feature matrix
    assignments : 2d-array
        Centroid assignments
    centroids : sequence
        Centroids
    model : str, optional
        one of

        'one_variance'
            All features share the same variance parameter sigma^2. Default

        'full_covariance'
            Estimate a full covariance matrix or use covs[i] for centroid[i]
    covs : sequence, optional
        Covariance matrices. If None, then estimate from the data. If scalars
        instead of matrices are given, then s stands for sI (i.e., the diagonal
        matrix with s along the diagonal).

    Returns
    -------
    B : float
        AIC value

    See Also
    --------
    BIC
    '''
    return _compute('AIC', fmatrix, assignments, centroids, model, covs)

