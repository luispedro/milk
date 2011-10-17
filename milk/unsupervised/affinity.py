# -*- coding: utf-8 -*-
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# Copyright (C) 2010-2011,
#       Luis Pedro Coelho <luis@luispedro.org>,
#       Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#       Gael Varoquaux <gael.varoquaux@normalesup.org>
#
# License: MIT. See COPYING.MIT file in the milk distribution
"""Affinity propagation

Original Authors (for scikits.learn):
        Alexandre Gramfort alexandre.gramfort@inria.fr
        Gael Varoquaux gael.varoquaux@normalesup.org

Luis Pedro Coelho made the implementation more careful about allocating
intermediate arrays.
"""

import numpy as np

__all__ = [
    'affinity_propagation',
    ]

def affinity_propagation(S, p=None, convit=30, maxit=200, damping=0.5, copy=True, R=0):
    """Perform Affinity Propagation Clustering of data

    Parameters
    ----------
    S : array [n_points, n_points]
        Matrix of similarities between points
    p : array [n_points,] or float, optional
        Preferences for each point
    damping : float, optional
        Damping factor
    copy : boolean, optional
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency
    R : source of randomness

    Returns
    -------

    cluster_centers_indices : array [n_clusters]
        index of clusters centers

    labels : array [n_points]
        cluster labels for each point

    Notes
    -----
    See examples/plot_affinity_propagation.py for an example.

    Reference:
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007

    """
    if copy:
        # Copy the affinity matrix to avoid modifying it inplace
        S = S.copy()

    n_points = S.shape[0]

    assert S.shape[0] == S.shape[1]

    if p is None:
        p = np.median(S)

    if damping < 0.5 or damping >= 1:
        raise ValueError('damping must be >= 0.5 and < 1')

    random_state = np.random.RandomState(R)

    # Place preferences on the diagonal of S
    S.flat[::(n_points+1)] = p

    A = np.zeros((n_points, n_points))
    R = np.zeros((n_points, n_points)) # Initialize messages

    # Remove degeneracies
    noise = random_state.randn(n_points, n_points)
    typeinfo = np.finfo(S.dtype)
    noise *= typeinfo.tiny*100
    S += noise
    del noise

    # Execute parallel affinity propagation updates
    e = np.zeros((n_points, convit))

    ind = np.arange(n_points)

    for it in range(maxit):
        Aold = A.copy()
        Rold = R.copy()
        A += S

        I = np.argmax(A, axis=1)
        Y = A[ind, I]#np.max(A, axis=1)

        A[ind, I] = typeinfo.min

        Y2 = np.max(A, axis=1)
        R = S - Y[:, np.newaxis]

        R[ind, I[ind]] = S[ind, I] - Y2

        Rold *= damping
        R *= (1-damping)
        R += Rold

        # Compute availabilities
        Rd = R.diagonal().copy()
        np.maximum(R, 0, R)
        R.flat[::n_points+1] = Rd

        A = np.sum(R, axis=0)[np.newaxis, :] - R

        dA = np.diag(A)
        A = np.minimum(A, 0)

        A.flat[::n_points+1] = dA

        Aold *= damping
        A *= (1-damping)
        A += Aold

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convit] = E
        K = np.sum(E, axis=0)

        if it >= convit:
            se = np.sum(e, axis=1);
            unconverged = np.sum((se == convit) + (se == 0)) != n_points
            if (not unconverged and (K>0)) or (it==maxit):
                print "Converged after %d iterations." % it
                break
    else:
        print "Did not converge"

    I = np.where(np.diag(A+R) > 0)[0]
    K = I.size # Identify exemplars

    if K > 0:
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K) # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c==k)[0]
            j = np.argmax(np.sum(S[ii, ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        labels = np.empty((n_points, 1))
        cluster_centers_indices = None
        labels.fill(np.nan)

    return cluster_centers_indices, labels
