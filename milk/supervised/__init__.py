# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

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
