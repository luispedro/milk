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
import numpy
from numpy import array, zeros, sqrt, inf, empty
from ..utils import get_pyrandom
from .normalise import zscore

__all__ = ['kmeans','repeated_kmeans']

def _euclidean2(fmatrix, x, output=None):
    if output is None:
        output = np.zeros(N)
    try:
        from scipy import weave
        from scipy.weave import converters
        N,q = fmatrix.shape
        output[:] = 0
        code = '''
        for (int i = 0; i != N; ++i) {
            for (int j = 0; j != q; ++j) {
                output(i) += (fmatrix(i,j) - x(j))*(fmatrix(i,j)-x(j));
            }
        }
        '''
        weave.inline(
                code,
                ['fmatrix','N','q','x','output'],
                type_converters=converters.blitz)
        return output
    except:
        output[:] = ((fmatrix - x)**2).sum(1)
        return output

def _mahalabonis2(fmatrix, x, icov, output=None):
    diff = (fmatrix-x)
    icov = (icov)
    # The expression below seems to be faster than looping over the elements and summing 
    return dot(diff,dot(icov,diff.T)).diagonal()

def centroid_errors(fmatrix,assignments,centroids):
    errors = []
    for k in xrange(len(centroids)):
        errors.extend(fmatrix[assignments == k] - centroids[k])
    return array(errors)

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
    if distance == 'euclidean':
        distfunction=_euclidean2
    elif distance == 'seuclidean':
        fmatrix = zscore(fmatrix)
        distfunction=_euclidean2
    elif distance == 'mahalanobis':
        icov = kwargs.get('icov',None)
        if icov is None:
            covmat = kwargs.get('covmat',None)
            if covmat is None:
                covmat = cov(fmatrix.T)
            icov = linalg.inv(covmat)
        distfunction = lambda f,x, out: _mahalabonis2(f, x, icov, output=out)
    else:
        raise 'Distance argument unknown (%s)' % distance
    R = get_pyrandom(R)

    N,q = fmatrix.shape
    centroids = np.array(R.sample(fmatrix,K), fmatrix.dtype)

    prev = np.zeros(N,np.int32)
    assignments = np.zeros(N,np.int32)
    dists = np.zeros(N, fmatrix.dtype)
    ndists = np.zeros(N, fmatrix.dtype)
    for i in xrange(max_iter):
        assignments[:] = 0
        computed = False
        if distfunction is _euclidean2:
            try:
                from scipy import weave
                from scipy.weave import converters
                if fmatrix.flags['C_CONTIGUOUS'] and fmatrix.dtype in (np.float32,np.double):
                    if fmatrix.dtype == np.float32:
                        type = 'float'
                    else:
                        type = 'double'
                    code = '''
#line 129 "kmeans.py"
                    for (int i = 0; i != N; ++i) {
                        %(type)s dist = std::numeric_limits<%(type)s>::infinity();
                        %(type)s* cd = centroids;
                        for (int k = 0; k != K; ++k) {
                            %(type)s ndist = 0.0;
                            %(type)s* fd = fmatrix + i * q;
                            for (int d = 0; d != q; ++d) {
                                ndist += (*fd-*cd)*(*fd-*cd);
                                ++fd;
                                ++cd;
                            }
                            if (ndist < dist) {
                                assignments[i] = k;
                                dist = ndist;
                            }
                        }
                    }
                    ''' % { 'type' : type }
                    weave.inline(code, ['fmatrix','N','q','K','assignments','centroids'], headers=['<limits>'])
                else: 
                    code = '''
                    for (int i = 0; i != N; ++i) {
                        float dist = std::numeric_limits<float>::infinity();
                        for (int k = 0; k != K; ++k) {
                            float ndist = 0.0;
                            for (int d = 0; d != q; ++d) {
                                ndist += (fmatrix(i,d) - centroids(k,d))*(fmatrix(i,d) - centroids(k,d));
                            }
                            if (ndist < dist) {
                                assignments(i) = k;
                                dist = ndist;
                            }
                        }
                    }
                    '''
                    weave.inline(
                        code,
                        ['fmatrix','N','q','K','assignments','centroids'],
                        type_converters=converters.blitz)
                computed = True
            except:
                pass
        if not computed:
            dists[:] = np.inf
            for ci,C in enumerate(centroids):
                ndists = distfunction(fmatrix, C, output=ndists)
                try:
                    from scipy import weave
                    from scipy.weave import converters
                    code = '''
                    for (int i = 0; i != N; ++i) {
                        if (ndists(i) < dists(i)) {
                            assignments(i) = ci;
                            dists(i) = ndists(i);
                        }
                    }
                    '''
                    weave.inline(
                        code,
                        ['dists', 'ndists', 'N', 'assignments', 'ci'],
                        type_converters=converters.blitz)
                except Exception, e:
                    print 'scipy.weave.inline failed. Resorting to Python code (Exception was "%s")' % e
                    better = (ndists < dists)
                    assignments[better] = ci
                    dists[better] = ndists[better]
        if np.all(assignments == prev):
            break
        try:
            from scipy import weave
            from scipy.weave import converters
            centroids[:] = 0
            counts = np.zeros(K,np.uint32)
            code = '''
            for (int i = 0; i != N; ++i) {
                int c = assignments(i);
                ++counts(c);
                for (int j = 0; j != q; ++j) {
                    centroids(c,j) += fmatrix(i,j);
                }
            }
            for (int i = 0; i != K; ++i) {
                for (int j = 0; j != q; ++j) {
                    centroids(i,j) /= counts(i);
                }
            }
            '''
            weave.inline(
                    code,
                    ['fmatrix','centroids','N','q','K','assignments','counts'],
                    type_converters=converters.blitz)
        except Exception, e:
            print 'scipy.weave.inline failed. Resorting to Python code (Exception was "%s")' % e
            centroids = array([fmatrix[assignments == C].mean(0) for C in xrange(K)])
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
    best=+inf
    for i in xrange(iterations):
        A,C = kmeans(fmatrix, k, distance, max_iter=max_iter, R=R,**kwargs)
        rss = residual_sum_squares(fmatrix,A,C,distance,**kwargs)
        if rss < best:
            Ab,Cb = A,C
            best = rss
    return Ab,Cb

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
