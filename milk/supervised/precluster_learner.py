# -*- coding: utf-8 -*-
# Copyright (C) 2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from milk.unsupervised.kmeans import select_best_kmeans, assign_centroids
from .base import supervised_model
from milk import defaultlearner

class precluster_model(supervised_model):
    def __init__(self, centroids, base):
        self.centroids = centroids
        self.base = base

    def apply(self, features):
        histogram = assign_centroids(features, self.centroids, histogram=True, normalise=1)
        return self.base.apply(histogram)
        

class precluster_learner(object):
    '''
    This learns a classifier by clustering the input features
    '''
    def __init__(self, ks, base=None, R=None):
        if base is None:
            base = defaultlearner()
        self.ks = ks
        self.R = R
        self.base = base

    def set_option(self, k, v):
        if k in ('R', 'ks'):
            setattr(self, k, v)
        else:
            self.base.set_option(k,v)

    def train(self, features, labels, **kwargs):
        allfeatures = np.vstack(features)
        assignments, centroids = select_best_kmeans(allfeatures, self.ks, repeats=1, method="AIC", R=self.R)
        histograms = [assign_centroids(f, centroids, histogram=True, normalise=1) for f in features]
        base_model = self.base.train(histograms, labels, **kwargs)
        return precluster_model(centroids, base_model)

