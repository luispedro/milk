# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np

__all__ = [
    'set2binary_array',
    ]

class set2binary_array_model(object):
    def __init__(self, universe):
        self.universe = list(universe)

    def apply(self, elems):
        res = np.zeros(len(self.universe) + 1, bool)
        for e in elems:
            try:
                res[self.universe.index(e)] = True
            except :
                res[-1] = True
        return res

class set2binary_array(object):
    def train(self, features, labels, normalisedlabels=False):
        allfeatures = set()
        for f in features:
            allfeatures.update(f)
        return set2binary_array_model(allfeatures)
