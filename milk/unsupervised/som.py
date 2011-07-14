# -*- coding: utf-8 -*-
# Copyright (C) 2010, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np

from ..utils import get_pyrandom
from . import _som

def putpoints(grid, points, L=.2, radius=4, iterations=1, shuffle=True, R=None):
    '''
    putpoints(grid, points, L=.2, radius=4, iterations=1, shuffle=True, R=None)

    Feeds elements of `points` into the SOM `grid`

    Parameters
    ----------
    grid : ndarray
        Self organising map
    points : ndarray
        data to feed to array
    L : float, optional
        How much to influence neighbouring points (default: .2)
    radius : integer, optional
        Maximum radius of influence (in L_1 distance, default: 4)
    iterations : integer, optional
        Number of iterations
    shuffle : boolean, optional
        Whether to shuffle the points before each iterations
    R : source of randomness
    '''
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
    if shuffle:
        random = get_pyrandom(R)
    for i in xrange(iterations):
        if shuffle:
            random.shuffle(points)
        _som.putpoints(grid, points, L, radius)

def closest(grid, f):
    '''
    y,x = closest(grid, f)

    Finds the coordinates of the closest point in the `grid` to `f`

    ::

        y,x = \\argmin_{y,x} { || grid[y,x] - f ||^2 }

    Parameters
    ----------
    grid : ndarray of shape Y,X,J
        self-organised map
    f : ndarray of shape J
        point

    Returns
    -------
    y,x : integers
        coordinates into `grid`
    '''
    delta = grid - f
    delta **= 2
    delta = delta.sum(2)
    return np.unravel_index(delta.argmin(), delta.shape)


def som(data, shape, iterations=1000, L=.2, radius=4, R=None):
    '''
    grid = som(data, shape, iterations=1000, L=.2, radius=4, R=None):

    Self-organising maps

    Parameters
    ----------
    points : ndarray
        data to feed to array
    shape : tuple
        Desired shape of output. Must be 2-dimensional.
    L : float, optional
        How much to influence neighbouring points (default: .2)
    radius : integer, optional
        Maximum radius of influence (in L_1 distance, default: 4)
    iterations : integer, optional
        Number of iterations
    R : source of randomness

    Returns
    -------
    grid : ndarray
        Map
    '''
    R = get_pyrandom(R)
    d = data.shape[1]
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    grid = np.array(R.sample(data, np.product(shape))).reshape(shape + (d,))
    putpoints(grid, data, L=L, radius=radius, iterations=iterations, shuffle=True, R=R)
    return grid
