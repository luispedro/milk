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
import scipy
from ...utils import get_nprandom

__all__ = ['hoyer_sparse_nnmf']

def _norm2(x):
    return (x**2).sum()

def _solve_alpha(s,m,L2):
    sm = s-m
    s2 = _norm2(s)
    sm2 = _norm2(sm)
    m2 = _norm2(m)
    dot = (m*sm).sum()
    alpha = (-dot + np.sqrt(dot**2 - sm2*(m2-L2)))/sm2
    return alpha

def _project(x,L1,L2):
    '''
    Implement projection onto sparse space
    '''
    x = np.asanyarray(x)
    n = len(x)

    s = x + (L1 - x.sum())/n
    Z = np.zeros(n,bool)
    while True:
        m = (~Z) * L1/(n-Z.sum())
        alpha = _solve_alpha(s,m,L2)
        s = m + alpha * (s - m)
        negs = (s < 0)
        if not negs.any():
            return s
        Z |= negs
        s[Z] = 0
        c = (s.sum() - L1)/(n-Z.sum())
        s = s - c*(~Z)

def _L1for(s,x,L2):
    '''
    Solve for L1 in

    s = [ sqrt(n) - L1/sqrt(L2)] / [sqrt(n) - 1]
    '''
    L2 = np.sqrt(L2)
    sn = np.sqrt(len(x))
    return L2*s*((sn-1)-1)

def sparse_nnmf(V, r, sparsenessW = None, sparsenessH = None, max_iter=10000,R=None):
    '''
    W,H = hoyer.sparse_nnmf(V, r, sparsenessW = None, sparsenessH = None, max_iter=10000, R=None)

    Implement sparse nonnegative matrix factorisation.

    Reference:
    "Non-negative Matrix Factorisation with Sparseness Constraints"
    by Patrik Hoyer
    in Journal of Machine Learning Research 5 (2004) 1457--1469
    '''
        
    n,m = V.shape
    R = get_nprandom(R)
    mu_W = .15
    mu_H = .15
    eps = 1e-8
    W = R.standard_normal((n,r))**2
    H = R.standard_normal((r,m))**2

    def fixW():
        for i in xrange(r):
            col = W[:,i]
            L2 = _norm2(col)
            W[:,i] = _project(col,_L1for(sparsenessW,col,L2),L2)

    def fixH():
        for i in xrange(r):
            row = H[i,:]
            L2 = _norm2(row)
            H[i,:] = _project(row,_L1for(sparsenessH,col,L2),L2)

    if sparsenessW is not None: fixW()
    if sparsenessH is not None: fixH()
    for i in xrange(max_iter):
        if sparsenessW is not None:
            W -= mu_W * np.dot(np.dot(W,H)-V,H.T)
            fixW()
        else:
            updateW = np.dot(V,H.T)/(np.dot(W,np.dot(H,H.T))+eps)
            W *= updateW
        if sparsenessH is not None:
            H -= mu_H * np.dot(W.T,np.dot(W,H)-V)
            fixH()
        else:
            updateH = np.dot(W.T,V)/(np.dot(np.dot(W.T,W),H)+eps)
            H *= updateH
    return W,H

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
