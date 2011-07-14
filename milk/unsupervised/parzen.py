# -*- coding: utf-8 -*-
# Copyright (C) 2010, Luis Pedro Coelho <luis@luispedro.org>
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

def get_parzen_1class_rbf_loocv(features):
    '''
    f,fprime = get_parzen_1class_rbf_loocv(features)

    Leave-one-out crossvalidation value for 1-class Parzen window evaluator for
    features.

    Parameters
    ----------
    features : ndarray
        feature matrix

    Returns
    -------
    f : function: double -> double
        function which evaluates the value of a window value. Minize to get the
        best window value.
    fprime : function: double -> double
        function: df/dh
    '''
    from milk.unsupervised.pdist import pdist
    D2 = -pdist(features)
    n = len(features)
    sumD2 = D2.sum()
    D2.flat[::(n+1)] = -np.inf
    def f(h):
        D2h = D2 / (2.*h)
        np.exp(D2h, D2h)
        val = D2h.sum()
        return val/np.sqrt(2*h*np.pi)
    def fprime(h):
        D2h = D2 / (2.*h)
        D2h.flat[::(n+1)] = 1.
        D2h *= np.exp(D2h)
        val = D2h.sum() - D2h.trace()
        val /= np.sqrt(2*h*np.pi)
        return -1./(4*np.pi*h)*f(h) + val
    return f,fprime

def parzen(features, h):
    '''
    f = parzen(features, h)

    Parzen window smoothing

    Parameters
    ----------
    features : ndarray
        feature matrix
    h : double
        bandwidth

    Returns
    -------
    f : callable (double^N -> double)
        density function
    '''
    sum2 = np.array([np.dot(f,f) for f in features])
    N = len(features)
    beta = np.sqrt(2*h*np.pi)/N
    def f(x):
        dist = np.dot(features, -2*x)
        dist += sum2
        dist += np.dot(c,c)
        dist /= 2.*h
        np.exp(dist, dist)
        val = dist.sum()
        return val*beta
    return f

