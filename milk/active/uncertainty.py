# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
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

'''
Uncertainty
============

Implements uncertainty-based active learning strategies.

These are strategies that are based on querying those elements in the pool
which we are most uncertain about

Functions
----------
    * entropy
    * one_minus_max
'''

from __future__ import division
import numpy

def entropy(model, pool):
    '''
    entropies = entropy(model, pool)

    Returns the entropy of each classification output for
    members in the pool.
    '''
    def _entropy(labels, ps):
        H = 0.
        for p in ps:
            if p > 1e-9:
                H += p * np.log(p)
        return H
    return [_entropy(model.apply(u)) for u in pool]

def one_minus_max(model,pool):
    '''
    oneminus = one_minus_max(model,pool)

    oneminus[i] = 1 - max_L { P(pool_i == L) }

    Returns one minus the probability for the best label guess.
    '''
    def _minus1(labels, ps):
        return 1. - np.max(ps)
    return [_minus1(model.apply(u)) for u in pool]


