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

from __future__ import division
import numpy
import numpy as np
from milk.supervised.normalise import sample_to_2min
import milk.supervised.normalise


def test_zscore_normalise():
    I=milk.supervised.normalise.zscore_normalise()
    numpy.random.seed(1234)
    features = numpy.random.rand(20,100)
    L = numpy.zeros(100)
    model = I.train(features, L)
    transformed = np.array([model.apply(f) for f in features])
    assert np.all( transformed.mean(0)**2 < 1e-7 )
    assert np.all( np.abs(transformed.std(0) - 1) < 1e-3 )

        
def test_sample_to_2min():
    A = np.zeros(256, np.int32)
    def test_one(A):
        selected = sample_to_2min(A)
        ratios = []
        for l0 in set(A):
            for l1 in set(A):
                ratios.append( (A[selected] == l0).sum() / (A[selected] == l1).sum() )
        assert np.max(ratios) <= 2.001
    A[20:] = 1
    yield test_one, A

    A[21:] = 1
    yield test_one, A

    A[129:] = 2
    yield test_one, A

def test_sample_to_2min_list():
    from collections import defaultdict
    def count(xs):
        counts = defaultdict(int)
        for x in xs:
            counts[x] += 1
        return counts
    labels = ["A"]*8 + ["B"]*12 + ["C"]*16 + ["D"] * 24 + ["E"] * 1000
    selected = sample_to_2min(labels)
    before = count(labels)
    after = count(np.array(labels)[selected])
    assert max(after.values()) == min(before.values())*2


def test_interval_normalise():
    interval = milk.supervised.normalise.interval_normalise()
    np.random.seed(105)
    features = np.random.randn(100, 5)
    model = interval.train(features, features[0] > 0)
    transformed = np.array([model.apply(f) for f in features])
    assert np.allclose(transformed.min(0), -1)
    assert np.allclose(transformed.max(0), +1)

