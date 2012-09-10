# -*- coding: utf-8 -*-
# Copyright (C) 2008-2012, Luis Pedro Coelho <luis@luispedro.org>
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
__all__ = ['zscore']

def zscore(features, axis=0, inplace=False):
    """
    features = zscore(features, axis=0, inplace=False)

    Returns a copy of features which has been normalised to zscores 

    Parameters
    ----------
    features : ndarray
        2-D input array
    axis : integer, optional
    inplace : boolean, optional
        Whether to operate inline
    """
    if features.ndim != 2:
        raise('milk.unsupervised.zscore: Can only handle 2-D arrays')
    mu = features.mean(axis)
    sigma = np.std(features, axis)
    sigma[sigma == 0] = 1.
    if not inplace:
        features = features.copy()
    if axis == 0:
        features -= mu
        features /= sigma
    elif axis == 1:
        features -= mu[:,None]
        features /= sigma[:,None]
    return features

