# -*- coding: utf-8 -*-
# Copyright (C) 2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from .base import supervised_model, base_adaptor

class one_by_one_model(supervised_model):
    def __init__(self, models):
        self.models = models

    def apply(self, fs):
        result = []
        for ell,model in self.models.iteritems():
            if model.apply(fs):
                result.append(ell)
        return result


class one_by_one(base_adaptor):
    '''
    Implements 1-vs-all multi-label classifier by transforming a base (binary)
    classifier.

    Example
    -------

    features = [....]
    labels = [
        (0,),
        (1,2),
        (0,2),
        (0,3),
        (1,2,3),
        (2,0),
        ...
        ]
    learner = one_by_one(milk.defaultlearner())
    model = learner.train(features, labels)
    '''
    def train(self, features, labels, **kwargs):
        universe = set()
        for ls in labels:
            universe.update(ls)
        models = {}
        for ell in universe:
            contained = np.array([int(ell in ls) for ls in labels])
            models[ell] = self.base.train(features, contained, normalisedlabels=True)
        return one_by_one_model(models)

