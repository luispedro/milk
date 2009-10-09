# -*- coding: utf-8 -*-
# Copyright (C) 2008-2009, Luís Pedro Coelho <lpc@cmu.edu>
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
from numpy.linalg import det
import scipy.stats
from . classifier import normaliselabels

_TOLERANCE = 0
_SIGNIFICANCE_IN = .15
_SIGNIFICANCE_OUT = .15

def _sweep(A, k, flag):
    N,_ = A.shape
    Akk = A[k,k]
    B = np.zeros_like(A)
    try:
        from scipy import weave
        from scipy.weave import converters
        k = int(k)
        code = '''
#line 42 "featureselection.py"
        for (int i = 0; i != N; ++i) {
            for (int j = 0; j != N; ++j) {
                if (i == k) {
                    if (j == k) {
                        B(i,j) =  - 1./A(k,k);
                    } else {
                        B(i,j)=flag*A(i,j)/A(k,k);
                    }
                } else if (j == k) {
                    B(i,j)=flag*A(i,j)/A(k,k);
                } else { 
                    B(i,j)=A(i,j) - A(i,k)*A(k,j)/A(k,k);
                }
            }
        }
        '''
        weave.inline(
                code,
                ['A','B','k','Akk','flag','N'],
                type_converters=converters.blitz)
    except:
        for i in xrange(N):
            for j in xrange(N):
                if i == k:
                    if j == k:
                        B[i,j] = -1./Akk
                    else:
                        B[i,j] = flag*A[i,j]/Akk
                elif j == k:
                    B[i,j] = flag*A[i,j]/Akk
                else:
                    B[i,j] = A[i,j] - A[i,k]*A[k,j]/Akk
    return B

def sda(features,labels):
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

    assert len(features) == len(labels), 'milk.supervised.featureselection.sda: length of features not the same as length of labels'
    N, m = features.shape
    labels,labelsu = normaliselabels(labels)
    q = len(labelsu)

    mus = np.array([features[labels==i,:].mean(0) for i in xrange(q)])
    mu = features.mean(0)
    
    W = np.zeros((m,m))
    T = np.zeros((m,m))
    try:
        from scipy import weave
        from scipy.weave import converters
        code='''
#line 106 "featureselection.py"
        for (int i = 0; i != m; ++i) {
            for (int j = 0; j != m; ++j) {
                for (int n = 0; n != N; ++n) {
                    int g=labels(n);
                    W(i,j) += (features(n,i)-mus(g,i))*(features(n,j)-mus(g,j));
                    T(i,j) += (features(n,i)-mu(i))*(features(n,j)-mu(j));
                }
            }
        }
        '''
        weave.inline(
                code,
                ['N','m','W','T','features','mu','mus','labels'],
                type_converters=converters.blitz)
    except ImportError:
        import warnings
        warnings.warn('scipy.weave failed. Resorting to (slow) Python code')
        for i in xrange(m):
            for j in xrange(m):
                for n in xrange(N):
                    g = labels[n]
                    W[i,j] += (features[n,i]-mus[g,i])*(features[n,j]-mus[g,j])
                    T[i,j] += (features[n,i]-mu[i])*(features[n,j]-mu[j])
    ignoreidx = ( W.diagonal() == 0 )
    if ignoreidx.any():
        idxs, = np.where(~ignoreidx)
        F = sda(features[:,~ignoreidx],labels)
        return idxs[F]
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
            if PrF > _SIGNIFICANCE_OUT:
                #print 'removing ',k, 'V(k)', 1./V_m, 'Fremove', Fremove, 'df1', df1, 'df2', df2, 'PrF', PrF
                if k == last_enter_k:
                    # We are going into an infinite loop.
                    import warnings
                    warnings.warn('pyslic.featureselection.sda: infinite loop detected (maybe bug?).')
                    break
                W = _sweep(W,k,1)
                T = _sweep(T,k,1)
                continue
        ks = ( (W_d / D) > _TOLERANCE)
        if ks.any():
            V_m = V[ks].min()
            k, = np.where(V==V_m)
            k = k[0]
            Fenter = (N-p-q)/(q-1) * (1-V_m)/V_m
            df2 = N-p-q
            PrF = 1 - scipy.stats.f.cdf(Fenter,df1,df2)
            if PrF < _SIGNIFICANCE_IN:
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

def _rank(A,tol=1e-8):
    s = np.linalg.svd(A,compute_uv=0)
    return (s > tol).sum()

def linear_independent_features(featmatrix, labels = None):
    '''
    Returns the indices of a set of linearly independent features (columns).

    indices = linear_independent_features(features)
    '''
    independent = []
    R = _rank(featmatrix)
    i = 0
    offset = 0
    while i < featmatrix.shape[1]:
        R_ = _rank(np.delete(featmatrix,i,1))
        if R_ == R:
            featmatrix = np.delete(featmatrix,i,1)
            offset += 1
        else:
            independent.append(i+offset)
            i += 1
    return np.array(independent)

class featureselector(object):
    '''
    selector = featureselector(function)

    Returns a transformer which selects features according to
        selected_idxs = function(features,labels)
    '''
    def __init__(self, selector):
        self.selector = selector

    def train(self, features, labels):
        self.idxs = self.selector(features, labels)

    def apply(self, features):
        return features[self.idxs]

def sda_filter():
    return featureselector(sda)

