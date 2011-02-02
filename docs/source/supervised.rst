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

All learners have a ``train`` function which takes 2 at least arguments:
- features : sequence of features
- labels : sequence of labels

(They may take more parameters).

They return a *model* object, which has an ``apply`` function which takes a
single input and returns its label.

Note that there are always two objects: the learned and the model and they are
independent. Every time you call ``learner.train()`` you get a new model. This
is different from the typical interface where you first call ``train()`` and
later ``apply()`` (or equivalent names) on the same object. This is a better
interface because the type system protects you against calling ``apply()`` on
the wrong object and because it often the case that you want to learn several
models with the same learner. The only disadvantage is that the word
*classifier* can be used for both, so in the documentation, we always refer to
*models* and *classifiers.*

Both learners and models are pickle()able.

Composition and Defaults
------------------------

The style of milk involves many small objects,each providing one step of the
pipeline. For example:

1. remove NaNs and Infs from features
2. bring features to the [-1, 1] interval
3. feature selection by removing linearly dependent features and then SDA
4. one-vs-rest classifier based on a grid search for parameters for an svm
   classifier

To get this you can use::

    classifier = ctransforms(
                    chkfinite(),
                    interval_normalise(),
                    featureselector(linear_independent_features),
                    sda_filter(),
                    gridsearch(one_against_one(svm.svm_to_binary(svm.svm_raw())),
                                params={
                                    'C': 2.**np.arange(-9,5),
                                    'kernel': [svm.rbf_kernel(2.**i) for i in np.arange(-7,4)],
                                }
                                ))

As you can see, this is very flexible, but can be tedious. Therefore, milk
provides the above as a single function call: ``defaultclassifier()``


supervised Submodules
---------------------

- defaultclassifier: contains a default "good enough" classifier
- svm: related to SVMs
- adaboost: Adaboost
- randomforest: random forests
- grouped: contains objects to transform single object learners into group
  learners by voting
- multi: transforms binary learners into multi-class learners (1-vs-1 or
  1-vs-rest)
- featureselection: feature selection
- knn: k-nearest neighbours
- tree: decision tree learners

