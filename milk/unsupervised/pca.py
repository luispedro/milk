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
from numpy import linalg
from . import normalise
    
def pca(X, zscore=True):
    '''
    Y,V = pca(X, zscore=True)

    Principal Component Analysis

    Performs principal component analysis. Returns transformed
    matrix and principal components

    Parameters
    ----------

        * X: data matrix
        * zscore: whether to normalise to zscores (default: True)
    '''
    if zscore:
        X = normalise.zscore(X)
    C = np.cov(X.T)
    w,v = linalg.eig(C)
    Y = np.dot(v,X.T).T
    return Y,v

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
