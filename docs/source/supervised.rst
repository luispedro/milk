=========================
Supervised Classification
=========================

Supervised learning takes in both a set of *input features* and their
corresponding *labels* to produce a model which can then be fed an unknown
instance and produce a label for it.

Typical supervised learning models are SVMs and decision trees.

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

Learners
--------

All learners have a `train` function which takes 2 arguments:
    - features : sequence of features
    - labels : sequence of labels

(They may take more parameters).

They return a `model` object, which has an `apply` function which takes a
single input and returns its label.

Note that there are always two objects: the learned and the model and they are
independent. Every time you call learner.train() you get a new model. This is
different from the typical interface where you first call `train()` and later
`apply()` (or equivalent names) on the same object. This is a better interface
because the type system protects you against calling `apply()` on the wrong
object and because it often the case that you want to learn several models with
the same learner. The only disadvantage is that the word *classifier* can be
used for both, so in the documentation, we always refer to *models* and
*classifiers.*

Both learners and models are pickle()able.


supervised Submodules
---------------------

- defaultclassifier: contains a default "good enough" classifier
- svm: related to SVMs
- grouped: contains objects to transform single object learners into group
  learners by voting
- multi: transforms binary learners into multi-class learners (1-vs-1 or
  1-vs-rest)
- featureselection: feature selection
- knn: k-nearest neighbours
- tree: decision tree learners

