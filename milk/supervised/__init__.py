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

'''
milk.supervised

This hold the supervised classification modules:

Submodules
----------

- defaultclassifier: contains a default "good enough" classifier
- svm: related to SVMs
- grouped: contains objects to transform single object classifiers into group classifiers
    by voting
- multi: transforms binary classifiers into multi-class classifiers (1-vs-1 or 1-vs-rest)
- featureselection: feature selection
- knn: k-nearest neighbours
- tree: decision tree classifiers

Classifiers
-----------

All classifiers have a `train` function which takes 2 arguments:
    - features : sequence of features
    - labels : sequence of labels
They return a `model` object, which has an `apply` function which takes a
single input and returns its label.

Note that there are always two objects: the learned and the model and they are
independent. Every time you call learner.train() you get a new model.

Both classifiers and models are pickle()able.

Example
-------
::

    features = np.random.randn(100,20)
    features[:50] *= 2
    labels = np.repeat((0,1), 50)

    classifier = milk.defaultclassifier()
    model = classifier.train(features, labels)
    new_label = model.apply(np.random.randn(100))
    new_label2 = model.apply(np.random.randn(100)*2)
'''

from .defaultclassifier import defaultclassifier, svm_simple
from .classifier import normaliselabels
from .gridsearch import gridsearch
from .tree import tree_learner

__all__ = [
    'normaliselabels',
    'defaultclassifier',
    'svm_simple',
    'gridsearch',
    'tree_learner',
    ]
