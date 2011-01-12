====================
Using Random Forests
====================

If you are not familiar with random forests, in general, `Wikipedia
<http://en.wikipedia.org/wiki/Random_forest>`__ is a good place to start
reading. The current article deals only with how to use them in **milk**.

Random forests are *binary classifiers*, so you need to use a transformer to
turn them into multi-class learners if you have multi-class data.

::

    from milk.supervised import randomforest
    from milk.supervised.multi import one_against_one

    rf_learner = randomforest.rf_learner()
    learner = one_against_one(rf_learner)

This is just another learner type, which we can use to train a model::

    from milksets import wine
    features, labels = wine.load()
    model = learner.train(features, labels)

or to perform cross-validation::
    
    cmat,names, preds = milk.nfoldcrossvalidation(features, labels, classifier=learner, return_predictions=1)

If you have `milksets <milksets.html>`__ installed, you can try it on one of its datasets::

    from milksets import wine
    features, labels = wine.load()
    cmat,names, preds = milk.nfoldcrossvalidation(features, labels, classifier=learner, return_predictions=1)

We can finally plot the results (mapped to 2 dimensions using PCA):

.. plot:: ./../milk/demos/rf_wine_2d.py
    :include-source:

Colours indicate the classification output. A circle means that it matches the
underlying label, a cross that it was a mis-classification.

