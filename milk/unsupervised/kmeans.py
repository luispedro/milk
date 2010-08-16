# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010, Luis Pedro Coelho <lpc@cmu.edu>
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
from ..utils import get_pyrandom
from .normalise import zscore

__all__ = ['kmeans','repeated_kmeans']


def _mahalabonis2(fmatrix, x, icov):
    diff = fmatrix-x
    # The expression below seems to be faster than looping over the elements and summing 
    return np.dot(diff, np.dot(icov, diff.T)).diagonal()

def centroid_errors(fmatrix,assignments,centroids):
    errors = []
    for k in xrange(len(centroids)):
        errors.extend(fmatrix[assignments == k] - centroids[k])
    return np.array(errors)

def residual_sum_squares(fmatrix,assignments,centroids,distance='euclidean',**kwargs):
    if distance != 'euclidean':
        raise NotImplemented, "residual_sum_squares only implemented for 'euclidean' distance"
    return (centroid_errors(fmatrix,assignments,centroids)**2).sum()

def kmeans(fmatrix,K,distance='euclidean',max_iter=1000,R=None,**kwargs):
    '''
    assignments, centroids = kmean(fmatrix, K, distance='euclidean', R=None, icov=None, covmat=None)

    K-Means Clustering

    Parameters
    ==========
        * distance can be one of:
            - 'euclidean'   : euclidean distance (default)
            - 'seuclidean'  : standartised euclidean distance. This is equivalent to first normalising the features.
            - 'mahalanobis' : mahalanobis distance.
                This can make use of the following keyword arguments:
                    + 'icov' (the inverse of the covariance matrix), 
                    + 'covmat' (the covariance matrix)
                If neither is passed, then the function computes the covariance from the feature matrix
        * max_iter: Maximum number of iteration
    '''
    fmatrix = np.asanyarray(fmatrix)
    if distance == 'seuclidean':
        fmatrix = zscore(fmatrix)
        distance = 'euclidean'
    if distance == 'euclidean':
        def distfunction(fmatrix, cs):
            dists = np.dot(fmatrix, -2*cs.T)
            dists += np.array([np.dot(c,c) for c in cs])
            # For a distance, we'd need to add the fmatrix**2 components, but
            # it doesn't matter because we are going to perform an argmin() on
            # the result.
            return dists
    elif distance == 'mahalanobis':
        icov = kwargs.get('icov', None)
        if icov is None:
            covmat = kwargs.get('covmat', None)
            if covmat is None:
                covmat = cov(fmatrix.T)
            icov = linalg.inv(covmat)
        def distfunction(fmatrix, cs):
            return np.array([_mahalabonis2(fmatrix, c, icov) for c in cs])
    else:
        raise ValueError('Distance argument unknown (%s)' % distance)
    R = get_pyrandom(R)

    centroids = np.array(R.sample(fmatrix,K), fmatrix.dtype)
    prev = np.zeros(len(fmatrix), np.int32)
    for i in xrange(max_iter):
        dists = distfunction(fmatrix, centroids)
        assignments = dists.argmin(1)
        if np.all(assignments == prev):
            break
        empty = []
        for ci in xrange(K):
            where = (assignments == ci)
            count = where.sum()
            if count:
                mean = np.dot(fmatrix.T, where)
                mean /= count
                centroids[ci] = mean
            else:
                empty.append(ci)
        if empty:
            centroids = np.delete(centroids, empty)
            K = len(centroids)
        prev[:] = assignments
    return assignments, centroids

def repeated_kmeans(fmatrix,k,iterations,distance='euclidean',max_iter=1000,R=None,**kwargs):
    '''
    assignments,centroids = repeated_kmeans(fmatrix, k, repeats, distance='euclidean',max_iter=1000,**kwargs)

    Runs kmeans repeats times and returns the best result as evaluated according to distance

    @see kmeans
    '''
    if distance == 'seuclidean':
        fmatrix = zscore(fmatrix)
        distance = 'euclidean'
    if distance != 'euclidean':
        raise NotImplementedError, "repeated_kmeans is only implemented for 'euclidean' or 'seuclidean' distance"
    best = np.inf
    for i in xrange(iterations):
        A,C = kmeans(fmatrix, k, distance, max_iter=max_iter, R=R,**kwargs)
        rss = residual_sum_squares(fmatrix,A,C,distance,**kwargs)
        if rss < best:
            Ab,Cb = A,C
            best = rss
    return Ab,Cb

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
