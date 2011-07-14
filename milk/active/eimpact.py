# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010, Luis Pedro Coelho <luis@luispedro.org>
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
from ..supervised import svm
from ..supervised.classifier import ctransforms


def expected_impacts(D,labels,U):
    '''
    EIs = expected_impacts(D,labels,U)

    Compute Expected impact for each element of U

    Eis[i]:  P(label[i] == 1) * IMPACT(label[i] == 1) + P(label[i] == 0) * IMPACT(label[i] == 0)
    '''
    assert len(D) == len(labels), 'Nr of labeled examples should match lenght of labels vector'

    K = svm.rbf_kernel(20000)
    prob_classifier = ctransforms(svm.svm_raw(kernel=K,C=4),svm.svm_sigmoidal_correction())
    label_classifier = ctransforms(svm.svm_raw(kernel=K,C=4),svm.svm_binary())

    prob_classifier.train(D,labels)
    u_probs = prob_classifier(U)
    u_labels = (u_probs > .5)
    impacts = []
    for u,p in zip(U,u_probs):
        print len(impacts)
        label_classifier.train(numpy.vstack((D,u)),numpy.hstack((labels,[0])))
        u_labels_0 = label_classifier(U)

        label_classifier.train(numpy.vstack((D,u)),numpy.hstack((labels,[1])))
        u_labels_1 = label_classifier(U)

        e_impact = (1.-p)*(u_labels != u_labels_0).sum() + p*(u_labels != u_labels_1).sum()

        impacts.append(e_impact)
    return impacts

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
