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
__all__ = [
    'center',
    'zscore',
    ]
def _nanmean(arr, axis=None):
    nancounts = np.sum(~np.isnan(arr), axis=axis)
    return np.nansum(arr,axis=axis)/nancounts
def _nanstd(arr, axis=None):
    if axis == 1:
        return _nanstd(arr.T, axis=0)
    mu = _nanmean(arr,axis=axis)
    return np.sqrt(_nanmean((arr-mu)**2, axis=axis))


def zscore(features, axis=0, can_have_nans=True, inplace=False):
    """
    features = zscore(features, axis=0, can_have_nans=True, inplace=False)

    Returns a copy of features which has been normalised to zscores

    Parameters
    ----------
    features : ndarray
        2-D input array
    axis : integer, optional
        which axis to normalise (default: 0)
    can_have_nans : boolean, optional
        whether ``features`` is allowed to have NaNs (default: True)
    inplace : boolean, optional
        Whether to operate inline (i.e., potentially change the input array).
        Default is False

    Returns
    -------
    features : ndarray
        zscored version of features
    """
    if features.ndim != 2:
        raise('milk.unsupervised.zscore: Can only handle 2-D arrays')
    if not inplace:
        features = features.copy()
    if can_have_nans:
        mu = _nanmean(features, axis)
        sigma = _nanstd(features, axis)
    else:
        mu = features.mean(axis)
        sigma = np.std(features, axis)
    sigma[sigma == 0] = 1.
    if axis == 0:
        features -= mu
        features /= sigma
    elif axis == 1:
        features -= mu[:,None]
        features /= sigma[:,None]
    return features



def center(features, axis=0, can_have_nans=True, inplace=False):
    '''
    centered, mean = center(features, axis=0, inplace=False)

    Center data

    Parameters
    ----------
    features : ndarray
        2-D input array
    axis : integer, optional
        which axis to normalise (default: 0)
    can_have_nans : boolean, optional
        whether ``features`` is allowed to have NaNs (default: True)
    inplace : boolean, optional
        Whether to operate inline (i.e., potentially change the input array).
        Default is False

    Returns
    -------
    features : ndarray
        centered version of features
    mean : ndarray
        mean values
    '''
    if can_have_nans:
        meanfunction = _nanmean
    else:
        meanfunction = np.mean
    features = np.array(features, copy=(not inplace), dtype=float)
    mean = meanfunction(features, axis=axis)
    if axis == 0:
        features -= mean
    elif axis == 1:
        features -= mean[:,None]
    else:
        raise ValueError('milk.unsupervised.center: axis ∉ { 0, 1}')
    return features, mean

