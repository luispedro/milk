# -*- coding: utf-8 -*-
# Copyright (C) 2008, Luís Pedro Coelho <lpc@cmu.edu>
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
import numpy
try:
    import ncreduce
    _std=ncreduce.std
except:
    _std=numpy.std

__all__ = ['zscore','zscore_normalise','interval_normalise','chkfinite','icdf_normalise']

def zscore(features):
    """
    features = zscore(features)

    Returns a copy of features which has been normalised to zscores 
    """
    mu = features.mean(0)
    sigma = _std(features,0)
    sigma[sigma == 0] = 1
    return (features - mu) / sigma

class subtract_divide(object):
    def __init__(self,features=None):
        if features:
            self.train(features)
    def __call__(self,features):
        return (features - self.shift)/self.factor

class zscore_normalise(subtract_divide):
    '''
    Normalise to z-scores

    A preprocessor that normalises features to z scores.
    '''
    def __init__(self,features=None):
        subtract_divide.__init__(self,features)

    def train(self,features,labels):
        self.shift = features.mean(0)
        self.factor = _std(features,0)
        self.factor[self.factor== 0.] = 1 # This makes the division a null op.

class interval_normalise(subtract_divide):
    '''
    Linearly scale to the interval [-1,1] (per libsvm recommendation)

    '''
    def __init__(self,features=None):
        subtract_divide.__init__(self,features)

    def train(self,features,labels):
        D = features.max(0) - features.min(0)
        self.shift = features.mean(0) + D/2.
        self.factor = D/2.
        self.factor[self.factor== 0.]=1 # This makes the division a null op.

class chkfinite(object):
    '''
    Fill NaN & Inf values

    Replaces NaN & Inf values with zeros.
    '''
    __slots__ = []
    def __init__(self):
        pass

    def train(self,features,labels):
        pass

    def apply(self,features):
        nans=isnan(features)+isinf(features)
        if nans.any():
            features=features.copy()
            features[nans]=0
        return features

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
