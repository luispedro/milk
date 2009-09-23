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

def defaultclassifier(mode='medium'):
    '''
    C = defaultclassifier(mode='medium')

    Parameters
    -----------

        * mode: One of ('fast','medium','slow'). This defines the
        speed accuracy trade-off. It essentially defines how large the
        SVM parameter range is.
    '''
    from .classifier import ctransforms
    from .gridsearch import gridsearch
    from . import svm
    from .normalise import chkfinite, interval_normalise
    from .featureselection import sda_filter, featureselector, linear_independent_features
    from .multi import one_against_one
    assert mode in ('slow','medium','fast'), "milk.supervised.defaultclassifier: mode must be one of 'fast','slow','medium'."
    if mode == 'fast':
        c_range = np.arange(-2,4)
        sigma_range = np.arange(-2,3)
    elif mode == 'medium':
        c_range = np.arange(-2,4)
        sigma_range = np.arange(-4,4)
    else: # mode == 'slow'
        c_range = np.arange(-7,4)
        sigma_range = np.arange(-4,4)
    return ctransforms(
            chkfinite(),
            interval_normalise(),
            featureselector(linear_independent_features),
            sda_filter(),
            gridsearch(one_against_one(
                            lambda: ctransforms( \
                                svm.svm_raw(), \
                                svm.svm_binary() \
                                )),
                        params={
                            (0,'C'): 2.**c_range,
                            (0,'kernel'): [svm.rbf_kernel(2.**i) for i in sigma_range],
                        }
                        ))


# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
