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
from collections import defaultdict
import numpy as np
try:
    import ncreduce
    _std = ncreduce.std
except:
    _std = np.std

__all__ = [
    'zscore',
    'zscore_normalise',
    'interval_normalise',
    'chkfinite',
    'sample_to_2min',
]

def zscore(features):
    """
    features = zscore(features)

    Returns a copy of features which has been normalised to zscores 
    """
    mu = features.mean(0)
    sigma = _std(features,0)
    sigma[sigma == 0] = 1
    return (features - mu) / sigma

class subtract_divide_model(object):
    def __init__(self, shift, factor):
        factor[factor == 0] = 1 # This makes the division a null op.

        self.shift = shift
        self.factor = factor

    def apply(self, features):
        return (features - self.shift)/self.factor

class zscore_normalise(object):
    '''
    Normalise to z-scores

    A preprocessor that normalises features to z scores.
    '''

    def train(self,features,labels):
        shift = features.mean(0)
        factor = _std(features,0)
        return subtract_divide_model(shift, factor)

class interval_normalise(object):
    '''
    Linearly scale to the interval [-1,1] (per libsvm recommendation)

    '''
    def train(self,features,labels):
        D = features.ptp(0)
        shift = features.mean(0) + D/2.
        factor = D/2.
        return subtract_divide_model(shift, factor)


def sample_to_2min(labels):
    '''
    selected = sample_to_2min(labels)

    Select examples so that the ratio of size of the largest
    class to the smallest class is at most two (i.e.,
        min_label_count = min { (labels == L).sum() | for L in set(labels) }
        for L' in set(labels):
            assert (labels == L').sum() <= 2 * min_label_count
    )

    Parameters
    ----------
        * labels: sequence of labels

    Output
    ------
        * selected: a Boolean numpy.ndarray
    '''
    counts = defaultdict(int)
    for n in labels:
        counts[n] += 1

    labels = np.asanyarray(labels)
    max_entries = np.min(counts.values())*2
    selected = np.zeros(len(labels), bool)
    for c in counts.iterkeys():
        p, = np.where(labels == c)
        p = p[:max_entries]
        selected[p] = 1
    return selected



class chkfinite(object):
    '''
    Fill NaN & Inf values

    Replaces NaN & Inf values with zeros.
    '''
    def __init__(self):
        pass

    def train(self,features,labels):
        return self

    def apply(self, features):
        nans = np.isnan(features) + np.isinf(features)
        if nans.any():
            features = features.copy()
            features[nans] = 0
        return features


# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
