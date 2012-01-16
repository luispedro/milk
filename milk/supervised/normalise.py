# -*- coding: utf-8 -*-
# Copyright (C) 2008-2012, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from .base import supervised_model

__all__ = [
    'zscore',
    'zscore_normalise',
    'interval_normalise',
    'chkfinite',
    'sample_to_2min',
    'normaliselabels'
]

def zscore(features):
    """
    features = zscore(features)

    Returns a copy of features which has been normalised to zscores 
    """
    mu = features.mean(0)
    sigma = np.std(features,0)
    sigma[sigma == 0] = 1
    return (features - mu) / sigma

class subtract_divide_model(object):
    def __init__(self, shift, factor):
        factor[factor == 0] = 1 # This makes the division a null op.

        self.shift = shift
        self.factor = factor

    def apply(self, features):
        return (features - self.shift)/self.factor

    def __repr__(self):
        return 'subtract_divide_model(%s, %s)' % (self.shift, self.factor)

class zscore_normalise(object):
    '''
    Normalise to z-scores

    A preprocessor that normalises features to z scores.
    '''

    def train(self, features, labels, **kwargs):
        shift = features.mean(0)
        factor = np.std(features,0)
        return subtract_divide_model(shift, factor)

class interval_normalise(object):
    '''
    Linearly scale to the interval [-1,1] (per libsvm recommendation)

    '''
    def train(self, features, labels, **kwargs):
        ptp = features.ptp(0)
        shift = features.min(0) + ptp/2.
        factor = ptp/2.
        return subtract_divide_model(shift, factor)

    def __repr__(self):
        return 'interval_normalise()'


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
    labels : sequence of labels

    Returns
    -------
    selected : a Boolean numpy.ndarray
    '''
    from collections import defaultdict
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



class chkfinite(supervised_model):
    '''
    Fill NaN & Inf values

    Replaces NaN & Inf values with zeros.
    '''
    def __init__(self):
        pass

    def train(self, features, labels, **kwargs):
        return self

    def apply(self, features):
        nans = np.isnan(features) + np.isinf(features)
        if nans.any():
            features = features.copy()
            features[nans] = 0
        return features

    def __repr__(self):
        return 'chkfinite()'

def normaliselabels(labels, multi_label=False):
    '''
    normalised, names = normaliselabels(labels, multi_label=False)

    If not ``multi_label`` (the default), normalises the labels to be integers
    from 0 through N-1. Otherwise, assume that each label is actually a
    sequence of labels.

    ``normalised`` is a np.array, while ``names`` is a list mapping the indices to
    the old names.

    Parameters
    ----------
    labels : any iterable of labels
    multi_label : bool, optional
        Whether labels are actually composed of multiple labels

    Returns
    ------
    normalised : a numpy ndarray
        If not ``multi_label``, this is an array of integers 0 .. N-1;
        otherwise, it is a boolean array of size len(labels) x N
    names : list of label names
    '''
    if multi_label:
        names = set()
        for ell in labels: names.update(ell)
        names = list(sorted(names))
        normalised = np.zeros( (len(labels), len(names)), bool)
        for i,ls in enumerate(labels):
            for ell in map(names.index, ls):
                normalised[i,ell] = True
        return normalised, names
    else:
        names = sorted(set(labels))
        normalised = map(names.index, labels)
        return np.array(normalised), names

