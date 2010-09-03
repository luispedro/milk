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
import numpy as np
from numpy import log, pi, array
from numpy.linalg import det, inv
from kmeans import residual_sum_squares, centroid_errors
import scipy

__all__ = ['BIC','AIC','log_likelihood','nr_parameters']

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
                covm = cov(diff.T)
            else:
                covm = covs[k]
            if covm.shape == ():
                covm = mat([[covm]])
            icov = mat(inv(covm))
            diff = mat(diff)
            Nk = diff.shape[0]
            res += -Nk/2.*log(det(covm)) + \
                 -.5 * (diff * icov * diff.T).diagonal().sum() 
        return res

    raise ValueError, "log_likelihood: cannot handle model '%s'" % model

    
def nr_parameters(fmatrix,k,model='one_variance'):
    N,q = fmatrix.shape
    if model == 'one_variance':
        return k*q+1
    elif model == 'diagonal_covariance':
        return k*(q+1)
    elif model == 'full_covariance':
        return k*+q*q

    raise ValueError, "nr_parameters: cannot handle model '%s'" % model

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
      fmatrix : feature matrix
      assignments : Centroid assignments
      centroids : Centroids
      model : one of:
                'one_variance': All features share the same variance parameter
                    sigma^2. Default
                'full_covariance': Estimate a full covariance matrix or use
                    covs[i] for centroid[i]
      covs : Covariance matrices
    Returns
    -------
      B = BIC value

    See Also
    --------
     `AIC`
    '''
    return _compute('BIC', fmatrix, assignments, centroids, model, covs)

def AIC(fmatrix,assignments,centroids,model='one_variance',covs=None):
    '''
    A = AIC(fmatrix,assignments,centroids,model)

    Compute Akaike Information Criterion

    Parameters
    ----------
      fmatrix : feature matrix
      assignments : Centroid assignments
      centroids : Centroids
      model : one of:
                'one_variance': All features share the same variance parameter
                    sigma^2. Default
                'full_covariance': Estimate a full covariance matrix or use
                    covs[i] for centroid[i]
      covs : Covariance matrices
    Returns
    -------
      A = AIC value

    See Also
    --------
     `BIC`
    '''
    return _compute('AIC', fmatrix, assignments, centroids, model, covs)

