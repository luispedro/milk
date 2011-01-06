# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np

def get_parzen_rbf_loocv(features,labels):
    xij = np.dot(features,features.T)
    f2 = np.sum(features**2,1)
    d = f2-2*xij
    d = d.T + f2
    d_argsorted = d.argsort(1)
    d_sorted = d.copy()
    d_sorted.sort(1)
    e_d = np.exp(-d_sorted)
    labels_sorted = labels[d_argsorted].astype(np.double)
    labels_sorted *= 2
    labels_sorted -= 1
    def f(sigma):
        k = e_d ** (1./sigma)
        return (((k[:,1:] * labels_sorted[:,1:]).sum(1) > 0) == labels).mean()
    return f


