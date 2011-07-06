# -*- coding: utf-8 -*-
# Copyright (C) 2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division

class supervised_model(object):
    def apply_many(self, fs):
        '''
        labels = model.apply_many( examples )

        This is equivalent to ``map(model.apply, examples)`` but may be
        implemented in a faster way.

        Parameters
        ----------
        examples : sequence of training examples

        Returns
        -------
        labels : sequence of labels
        '''
        return map(self.apply, fs)


class base_adaptor(object):
    def __init__(self, base):
        self.base = base

    def set_option(self, k, v):
        self.base.set_option(k, v)
