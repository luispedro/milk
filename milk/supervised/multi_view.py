# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

import numpy as np


__all__ = [
    'multi_view_learner',
    ]
class multi_view_model(object):
    def __init__(self, models):
        self.models = models

    def apply(self, features):
        if len(features) != len(self.models):
            raise ValueError('milk.supervised.two_view: Nr of features does not match training data (got %s, expected %s)' % (len(features) ,len(self.models)))
        Ps = np.array([model.apply(f) for model,f in zip(self.models, features)])
        if np.any(Ps <= 0.): return False
        if np.any(Ps >= 1.): return True
        # This is binary only:
        # if \prod Pi > \prod (1-Pi) return 1
        # is equivalent to
        # if \prod Pi/(1-Pi) > 1. return 1
        # if \sum \log( Pi/(1-Pi) ) > 0. return 1
        return np.sum( np.log(Ps/(1-Ps)) ) > 0


class multi_view_learner(object):
    '''
    Multi View Learner

    This learner learns different classifiers on multiple sets of features and
    combines them for classification.

    '''
    def __init__(self, bases):
        self.bases = bases

    def train(self, features, labels, normalisedlabels=False):
        features = zip(*features)
        if len(features) != len(self.bases):
            raise ValueError('milk.supervised.multi_view_learner: ' +
                        'Nr of features does not match classifiser construction (got %s, expected %s)'
                        % (len(features) ,len(self.bases)))
        models = []
        for basis,f in zip(self.bases, features):
            try:
                f = np.array(f)
            except:
                f = np.array(f, dtype=object)
            models.append(basis.train(f, labels))
        return multi_view_model(models)

multi_view_classifier = multi_view_learner
