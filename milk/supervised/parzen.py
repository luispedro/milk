# -*- coding: utf-8 -*-
# Copyright (C) 2008-2009, Luís Pedro Coelho <lpc@cmu.edu>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from __future__ import division
import numpy as np

def slow_parzen(features, labels, sigma):
    correct = 0
    N = len(features)
    labels = 2*labels - 1
    def kernel(fi, fj):
        return np.exp(-((fi-fj)**2).sum()/sigma)
    for i in xrange(N):
        C = 0.
        for j in xrange(N):
            if i == j: continue
            C += labels[j] * kernel(features[i],features[j])
        if (C*labels[i] > 0): correct += 1
    return correct/N

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


# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
