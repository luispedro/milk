==============================
MILK: MACHINE LEARNING TOOLKIT
==============================
Machine Learning in Python
--------------------------

Milk is a machine learning toolkit in Python.

Its focus is on supervised classification with several classifiers available:
SVMs (based on libsvm), k-NN, random forests, decision trees. It also performs
feature selection. These classifiers can be combined in many ways to form
different classification systems.

For unsupervised learning, milk supports k-means clustering and affinity
propagation.

Milk is flexible about its inputs. It optimised for numpy arrays, but can often
handle anything (for example, for SVMs, you can use any dataype and any kernel
and it does the right thing).

There is a strong emphasis on speed and low memory usage. Therefore, most of
the performance sensitive code is in C++. This is behind Python-based
interfaces for convenience.

To learn more, check the docs at `http://packages.python.org/milk/
<http://packages.python.org/milk/>`_ or the code demos included with the source
at ``milk/demos/``.

Examples
--------

Here is how to test how well you can classify some ``features,labels`` data,
measured by cross-validation::

    import numpy as np
    import milk
    features = np.random.rand(100,10) # 2d array of features: 100 examples of 10 features each
    labels = np.zeros(100)
    features[50:] += .5
    labels[50:] = 1
    confusion_matrix, names = milk.nfoldcrossvalidation(features, labels)
    print 'Accuracy:', confusion_matrix.trace()/float(confusion_matrix.sum())

If want to use a classifier, you instanciate a *learner object* and call its
``train()`` method::

    import numpy as np
    import milk
    features = np.random.rand(100,10)
    labels = np.zeros(100)
    features[50:] += .5
    labels[50:] = 1
    learner = milk.defaultclassifier()
    model = learner.train(features, labels)

    # Now you can use the model on new examples:
    example = np.random.rand(10)
    print model.apply(example)
    example2 = np.random.rand(10)
    example2 += .5
    print model.apply(example2)
    
There are several classification methods in the package, but they all use the
same interface: ``train()`` returns a *model* object, which has an ``apply()``
method to execute on new instances.


Details
-------
License: MIT

Author: Luis Pedro Coelho (with code from LibSVM and scikits.learn)

API Documentation: `http://packages.python.org/milk/ <http://packages.python.org/milk/>`_

Mailing List: `http://groups.google.com/group/milk-users
<http://groups.google.com/group/milk-users>`__

Features
--------
- SVMs. Using the libsvm solver with a pythonesque wrapper around it.
- K-means using as little memory as possible. It can cluster millions of
  instances efficiently.
- Random forests
- Self organising maps
- Stepwise Discriminant Analysis for feature selection.
- Non-negative matrix factorisation
- Affinity propagation

Recent History
--------------

The ChangeLog file contains a more complete history.


New in 0.4.2 (16 Jan 2012)
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Make defaultlearner able to take extra arguments
- Make ctransforms_model a supervised_model (adds apply_many)
- Add expanded argument to defaultlearner
- Fix corner case in SDA
- Fix repeated_kmeans
- Fix parallel gridminimise on Windows
- Add multi_label argument to normaliselabels
- Add multi_label argument to nfoldcrossvalidation.foldgenerator
- Do not fork a process in gridminimise if nprocs == 1 (makes for easier
  debugging, at the cost of slightly more complex code).
- Add milk.supervised.multi_label
- Fix ext.jugparallel when features is a Task
- Add milk.measures.bayesian_significance


New in 0.4.1
~~~~~~~~~~~~
- Fix important bug in multi-process gridsearch

New in 0.4.0
~~~~~~~~~~~~
- Use multiprocessing to take advantage of multi core machines (off by
  default).
- Add perceptron learner
- Set random seed in random forest learner
- Add warning to milk/__init__.py if import fails
- Add return value to ``gridminimise``
- Set random seed in ``precluster_learner``
- Implemented Error-Correcting Output Codes for reduction of multi-class
  to binary (including probability estimation)
- Add ``multi_strategy`` argument to ``defaultlearner()``
- Make the dot kernel in svm much, much, faster
- Make sigmoidal fitting for SVM probability estimates faster
- Fix bug in randomforest (patch by Wei on milk-users mailing list)

New in 0.3.10
~~~~~~~~~~~~~
- Add ext.jugparallel for integration with `jug <http://luispedro.org/software/jug>`_
- parallel nfold crossvalidation using jug
- parallel multiple kmeans runs using jug
- cluster_agreement for non-ndarrays
- Add histogram & normali(z|s)e options to ``milk.kmeans.assign_centroid``
- Fix bug in sda when features were constant for a class
- Add select_best_kmeans
- Added defaultlearner as a better name than defaultclassifier
- Add ``measures.curves.precision_recall``
- Add ``unsupervised.parzen.parzen``

New in 0.3.9
~~~~~~~~~~~~
- Add ``folds`` argument to ``nfoldcrossvalidation``
- Add ``assign_centroid`` function in milk.unsupervised.nfoldcrossvalidation
- Improve speed of k-nearest neighbour (10x on scikits-learn benchmark)
- Improve kmeans on newer numpy (works for larger datasets too)
- Faster kmeans by coding centroid recalculation in C++
- Fix gridminize for low count labels
- Fix bug with non-integer labels for tree learning

New in 0.3.8
~~~~~~~~~~~~
- Fix compilation on Windows

New in 0.3.7
~~~~~~~~~~~~
- Logistic regression
- Source demos included (in source and documentation)
- Add cluster agreement metrics
- Fix nfoldcrossvalidation bug when using origins

New in 0.3.6
~~~~~~~~~~~~
- Unsupervised (1-class) kernel density modeling
- Fix for when SDA returns empty
- weights option to some learners
- stump learner
- Adaboost (result of above changes)

