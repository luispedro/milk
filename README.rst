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

New in 0.6.1 (11 May 2015)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Fixed source distribution

New in 0.6 (27 Apr 2015)
~~~~~~~~~~~~~~~~~~~~~~~~
- Update for Python 3

New in 0.5.3 (19 Jun 2013)
~~~~~~~~~~~~~~~~~~~~~~~~~
- Fix MDS for non-array inputs
- Fix MDS bug
- Add return_* arguments to kmeans
- Extend zscore() to work on non-ndarrays
- Add frac_precluster_learner
- Work with older C++ compilers


New in 0.5.2 (7 Mar 2013)
~~~~~~~~~~~~~~~~~~~~~~~~~
- Fix distribution of Eigen with source

New in 0.5.1 (11 Jan 2013)
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Add subspace projection kNN
- Export ``pdist`` in milk namespace
- Add Eigen to source distribution
- Add measures.curves.roc
- Add ``mds_dists`` function
- Add ``verbose`` argument to milk.tests.run


New in 0.5 (05 Nov 2012)
~~~~~~~~~~~~~~~~~~~~~~~~
- Add coordinate-descent based LASSO
- Add unsupervised.center function
- Make zscore work with NaNs (by ignoring them)
- Propagate apply_many calls through transformers
- Much faster SVM classification with means a much faster defaultlearner()
  [measured 2.5x speedup on yeast dataset!]


For older versions, see ``ChangeLog`` file
