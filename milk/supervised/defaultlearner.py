# -*- coding: utf-8 -*-
# Copyright (C) 2008-2012, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np

__all__ = [
    'defaultlearner',
    'svm_simple',
    'feature_selection_simple',
    ]

def defaultlearner(mode='medium', multi_strategy='1-vs-1', expanded=False):
    '''
    learner = defaultlearner(mode='medium')

    Return the default classifier learner

    This is an SVM based classifier using the 1-vs-1 technique for multi-class
    problems (by default, see the ``multi_strategy`` parameter). The features
    will be first cleaned up (normalised to [-1, +1]) and go through SDA
    feature selection.

    Parameters
    -----------
    mode : string, optional
        One of ('fast','medium','slow', 'really-slow'). This defines the speed
        accuracy trade-off. It essentially defines how large the SVM parameter
        range is.
    multi_strategy : str, optional
        One of ('1-vs-1', '1-vs-rest', 'ecoc'). This defines the strategy used
        to convert the base binary classifier to a multi-class classifier.
    expanded : boolean, optional
        If true, then instead of a single learner, it returns a list of
        possible learners.

    Returns
    -------
    learner : classifier learner object or list
        If `expanded`, then it returns a list

    See Also
    --------
    feature_selection_simple : Just perform the feature selection
    svm_simple : Perform classification
    '''
    # These cannot be imported at module scope!
    # The reason is that they introduce a dependency loop:
    # gridsearch depends on nfoldcrossvalidation
    #   nfoldcrossvalidation depends on defaultlearner
    #   which cannot depend on gridsearch
    #
    # Importing at function level keeps all these issues at bay
    #
    from .classifier import ctransforms
    from .gridsearch import gridsearch
    from . import svm
    from .normalise import chkfinite, interval_normalise
    from .featureselection import sda_filter, featureselector, linear_independent_features
    from .multi import one_against_one, one_against_rest, ecoc_learner

    assert mode in ('really-slow', 'slow', 'medium', 'fast'), \
        "milk.supervised.defaultlearner: mode must be one of 'fast','slow','medium'."
    if multi_strategy == '1-vs-1':
        multi_adaptor = one_against_one
    elif multi_strategy == '1-vs-rest':
        multi_adaptor = one_against_rest
    elif multi_strategy == 'ecoc':
        multi_adaptor = ecoc_learner
    else:
        raise ValueError('milk.supervised.defaultlearner: Unknown value for multi_strategy: %s' % multi_strategy)

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

    kernels = [svm.rbf_kernel(2.**i) for i in sigma_range]
    Cs = 2.**c_range

    if expanded:
        return [ctransforms(feature_selection_simple(),
                    multi_adaptor(svm.svm_to_binary(svm.svm_raw(C=C, kernel=kernel))))
                    for C in Cs for kernel in kernels]
    return ctransforms(feature_selection_simple(),
            gridsearch(multi_adaptor(svm.svm_to_binary(svm.svm_raw())),
                        params={ 'C': Cs, 'kernel': kernels, }))


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
    defaultlearner : perform feature selection *and* classification
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
    defaultlearner : feature selection and gridsearch for SVM parameters
    '''
    from . import svm
    from .multi import one_against_one
    return one_against_one(svm.svm_to_binary(svm.svm_raw(C=C, kernel=kernel)))

