# -*- coding: utf-8 -*-
# Copyright (C) 2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np

def precision_recall(values, labels, mode='all', nr_steps=100):
    '''
    precision, recall = precision_recall(values, labels, mode='all', nr_steps=100)

    Compute a precision-recall curve.

    For a given threshold ``T``, consider that the positions where ``values >=
    T`` are classified as True. Precision is defined as ``TP/(TP+FP)``, while
    recall is defined as ``TP/(TP+FN)``.

    Parameters
    ----------
    values : sequence of numbers
    labels : boolean sequence
    mode : str, optional
        Which thresholds to consider. Either 'all' (i.e., use all values of
        `values` as possible thresholds), or 'step' (using `nr_steps`
        equidistant points from ``min(values)`` to ``max(values)``)
    nr_steps : integer, optional
        How many steps to use. Only meaningfule if ``mode == 'steps'``

    Returns
    -------
    precision : a sequence of floats
    recall : a sequence of floats

    Actually, ``2 x P`` array is returned.
    '''

    values = np.asanyarray(values)
    labels = np.asanyarray(labels)
    if len(values) != len(labels):
        raise ValueError('milk.measures.precision_recall: `values` must be of same length as `labels`')
    if mode == 'all':
        points = list(set(values))
        points.sort()
    elif mode == 'steps':
        points = np.linspace(values.min(), values.max(), nr_steps)
    else:
        raise ValueError('milk.measures.precision_recall: cannot handle mode: `%s`' % mode)
    true_pos = float(np.sum(labels))
    precision_recall = np.empty((len(points),2), np.float)

    for i,p in enumerate(points):
        selected = (values >= p)
        selected = labels[selected]
        precision_recall[i] = (np.mean(selected), np.sum(selected)/true_pos)
    return precision_recall.T

