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
- LASSO
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


New in 0.5 (05 Nov 2012)
~~~~~~~~~~~~~~~~~~~~~~~~
- Add coordinate-descent based LASSO
- Add unsupervised.center function
- Make zscore work with NaNs (by ignoring them)
- Propagate apply_many calls through transformers
- Much faster SVM classification with means a much faster defaultlearner()
  [measured 2.5x speedup on yeast dataset!]


New in 0.4.3 (17 Sept 2012)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Add select_n_best & rank_corr to featureselection
- Add Euclidean MDS
- Add tree multi-class strategy
- Fix adaboost with boolean weak learners (issue #6, reported by audy
  (Austin Richardson))
- Add ``axis`` arguments to zscore()


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

For older versions, see ``ChangeLog`` file
