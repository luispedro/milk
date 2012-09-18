# -*- coding: utf-8 -*-
# Copyright (C) 2008-2012, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

'''
milk.unsupervised

Unsupervised Learning
---------------------

- kmeans: This is a highly optimised implementation of kmeans
- PCA: Simple implementation
- Non-negative matrix factorisation: both direct and with sparsity constraints
'''

from kmeans import kmeans,repeated_kmeans, select_best_kmeans
from gaussianmixture import *
from pca import pca, mds
import nnmf
from nnmf import *
from pdist import pdist, plike
from .som import som
from .normalise import zscore

__all__ = [
    'kmeans',
    'repeated_kmeans',
    'select_best_kmeans',
    'pca',
    'pdist',
    'plike',
    'mds'
    'som',
    'zscore',
    ] + \
    nnmf.__all__
