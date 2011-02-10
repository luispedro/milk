========
AdaBoost
========

Adaboost
--------

This example is available as part of milk as ``milk/demos/adaboost.py``.

Adaboost is based on a weak learner. For this example, we are going to use a
stump learner::

    import milk.supervised.tree
    import milk.supervised.adaboost

    weak = milk.supervised.tree.stump_learner()
    learner = milk.supervised.adaboost.boost_learner(weak)

Currently, only binary classification is implemented for ``boost_learner``.
Therefore, we need to use a converter, in this case, using the *one versus one*
strategy::

    import milk.supervised.multi
    learner = milk.supervised.multi.one_against_one(learner)

Now, we can use this learner as we would normally do. For example, for
cross-validation::

    from milksets import wine
    features, labels = wine.load()
    cmat,names,predictions = \
        milk.nfoldcrossvalidation(features, \
                                    labels, \
                                    classifier=learner, \
                                    return_predictions=True)

We display just the first two dimensions here using circles for correct
predictions and crosses for mis-matches. The colour represents the underlying
class::

    import pylab as plt
    colors = "rgb"
    codes = "xo"
    for y,x,r,p in zip(features.T[0], features.T[1], labels, predictions):
        code = codes[int(r == p)]
        plt.plot([y],[x], colors[p]+code)
    plt.show()

.. plot:: ./../milk/demos/adaboost.py
    :include-source:

API Documentation
-----------------

.. automodule:: milk.supervised.adaboost
    :members: boost_learner

