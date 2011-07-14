# -*- coding: utf-8 -*-
# Copyright (C) 2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np

def rand_arand_jaccard(recovered, labels):
    '''
    rand, a_rand, jaccard = rand_arand_jaccard(recovered, labels)

    Compute Rand, Adjusted Rand, and Jaccard indices

    These share most of the computation. Therefore, it is best to compute them
    together even if you are only going to use some.

    Parameters
    ----------
    recovered : sequence of int
        The recovered clusters
    labels : sequence of int
        Underlying labels

    Returns
    -------
    rand : float
        Rand index
    a_rand : float
        Adjusted Rand index
    jaccard : float
        Jaccard index

    References
    ----------
    http://en.wikipedia.org/wiki/Rand_index
    http://en.wikipedia.org/wiki/Jaccard_index
    '''

    from scipy.misc import comb
    recovered = np.asanyarray(recovered)
    labels = np.asanyarray(labels)
    contig,_,_ = np.histogram2d(recovered, labels,np.arange(max(recovered.max()+2,labels.max()+2)))
    A_0 = contig.sum(0)
    A_1 = contig.sum(1)
    Ai2 = np.sum(A_0*(A_0-1)/2.)
    Bi2 = np.sum(A_1*(A_1-1)/2.)
    n = A_0.sum()

    a = comb(contig.ravel(), 2).sum()
    b = comb(A_0, 2).sum()-a
    c = comb(A_1, 2).sum()-a
    d = comb(n, 2)-a-b-c
    rand = (a+d)/(a+b+c+d)
    jaccard = (a+d)/(b+c+d)

    index = np.sum(contig*(contig-1)/2)
    expected = Ai2*Bi2/n/(n-1)*2.
    maxindex = (Ai2+Bi2)/2.
    a_rand = (index-expected)/(maxindex-expected)

    return rand, a_rand, jaccard

