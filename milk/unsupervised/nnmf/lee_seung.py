# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010, Luis Pedro Coelho <luis@luispedro.org>
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
from numpy import dot
from ...utils import get_nprandom

__all__ = ['nnmf']

def nnmf(V, r, cost='norm2', max_iter=int(1e4), tol=1e-8, R=None):
    '''
    A,S = nnmf(X, r, cost='norm2', tol=1e-8, R=None)

    Implement Lee & Seung's algorithm

    Parameters
    ----------
    V : 2-ndarray
        input matrix
    r : integer
        nr of latent features
    cost : one of:
        'norm2' : minimise || X - AS ||_2 (default)
        'i-div' : minimise D(X||AS), where D is I-divergence (generalisation of K-L divergence)
    max_iter : integer, optional
        maximum number of iterations (default: 10000)
    tol : double
        tolerance threshold for early exit (when the update factor is with tol
        of 1., the function exits)
    R : integer, optional
        random seed

    Returns
    -------
    A : 2-ndarray
    S : 2-ndarray

    Reference
    ---------
    "Algorithms for Non-negative Matrix Factorization"
    by Daniel D Lee, Sebastian H Seung
    (available at http://citeseer.ist.psu.edu/lee01algorithms.html)
    '''
    # Nomenclature in the function follows lee & seung, while outside nomenclature follows 
    eps = 1e-8
    n,m = V.shape
    R = get_nprandom(R)
    W = R.standard_normal((n,r))**2
    H = R.standard_normal((r,m))**2
    for i in xrange(max_iter):
        if cost == 'norm2':
            updateH = dot(W.T,V)/(dot(dot(W.T,W),H)+eps)
            H *= updateH
            updateW = dot(V,H.T)/(dot(W,dot(H,H.T))+eps)
            W *= updateW
        elif cost == 'i-div':
            raise NotImplementedError,'I-Div not implemented in lee_seung.nnmf'
        if True or (i % 10) == 0:
            max_update = max(updateW.max(),updateH.max())
            if abs(1.-max_update) < tol:
                break
    return W,H

