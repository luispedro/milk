# -*- coding: utf-8 -*-
# Copyright (C) 2010, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np

from ..utils import get_pyrandom
from . import _som

def putpoints(grid, points, L=.2, iterations=1, R=None, radius=None):
    if radius is None:
        radius = 4
    if type(L) != float:
        raise TypeError("milk.unsupervised.som: L should be floating point")
    if type(radius) != int:
        raise TypeError("milk.unsupervised.som: radius should be an integer")
    if grid.dtype != np.float32:
        raise TypeError('milk.unsupervised.som: only float32 arrays are accepted')
    if points.dtype != np.float32:
        raise TypeError('milk.unsupervised.som: only float32 arrays are accepted')
    if len(grid.shape) == 2:
        grid = grid.reshape(grid.shape+(1,))
    random = get_pyrandom(R)
    for i in xrange(iterations):
        random.shuffle(points)
        _som.putpoints(grid, points, L, radius)
