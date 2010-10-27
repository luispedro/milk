# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010, Luis Pedro Coelho <lpc@cmu.edu>
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
import numpy as np

def defaultclassifier(mode='medium'):
    '''
    learner = defaultclassifier(mode='medium')

    Return the default classifier learner

    This is an SVM based classifier using the 1-vs-1 technique for multi-class
    problems. The features will be first cleaned up (normalised to [-1, +1])
    and go through SDA feature selection.

    Parameters
    -----------
    mode : string, optional
        One of ('fast','medium','slow', 'really-slow'). This defines the speed
        accuracy trade-off. It essentially defines how large the SVM parameter
        range is.

    Returns
    -------
    learner : classifier learner object

    See Also
    --------
    feature_selection_simple : Just perform the feature selection
    svm_simple : Perform classification
    '''
    # These cannot be imported at module scope!
    # The reason is that they introduce a dependency loop:
    # gridsearch depends on nfoldcrossvalidation
    #   nfoldcrossvalidation depends on defaultclassifier
    #   which cannot depend on gridsearch
    #
    # Importing at function level keeps all these issues at bay
    #
    from .classifier import ctransforms
    from .gridsearch import gridsearch
    from . import svm
    from .normalise import chkfinite, interval_normalise
    from .featureselection import sda_filter, featureselector, linear_independent_features
    from .multi import one_against_one

    assert mode in ('really-slow', 'slow', 'medium', 'fast'), \
        "milk.supervised.defaultclassifier: mode must be one of 'fast','slow','medium'."
    if mode == 'fast':
        c_range = np.arange(-2,4)
        sigma_range = np.arange(-2,3)
    elif mode == 'medium':
        c_range = np.arange(-2,4)
        sigma_range = np.arange(-4,4)
    elif mode == 'really-slow':
        c_range = np.arange(-4,10)
        sigma_range = np.arange(-7,7)
    else: # mode == 'slow'
        c_range = np.arange(-9,5)
        sigma_range = np.arange(-7,4)
    return ctransforms(
            chkfinite(),
            interval_normalise(),
            featureselector(linear_independent_features),
            sda_filter(),
            gridsearch(one_against_one(svm.svm_to_binary(svm.svm_raw())),
                        params={
                            'C': 2.**c_range,
                            'kernel': [svm.rbf_kernel(2.**i) for i in sigma_range],
                        }
                        ))


def feature_selection_simple():
    '''
    selector = feature_selection_simple()

    Standard feature normalisation and selection

    This fills in NaNs and Infs (to 0 and large numbers),  normalises features
    to [-1, +1] and uses SDA for feature selection.

    Returns
    -------
    selector : supervised learner

    See Also
    --------
    defaultclassifier : perform feature selection *and* classification
    '''
    from .classifier import ctransforms
    from .normalise import chkfinite, interval_normalise
    from .featureselection import sda_filter, featureselector, linear_independent_features
    return ctransforms(
            chkfinite(),
            interval_normalise(),
            featureselector(linear_independent_features),
            sda_filter(),
            )

def svm_simple(C, kernel):
    '''
    learner = svm_simple(C, kernel)

    Returns a one-against-one SVM based classifier with `C` and `kernel`

    Parameters
    ----------
    C : double
        C parameter
    kernel : kernel
        Kernel to use

    Returns
    -------
    learner : supervised learner

    See Also
    --------
    feature_selection_simple : Perform feature selection
    defaultclassifier : feature selection and gridsearch for SVM parameters
    '''
    from . import svm
    from .multi import one_against_one
    return one_against_one(svm.svm_to_binary(svm.svm_raw(C=C, kernel=kernel)))

