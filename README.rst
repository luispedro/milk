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

New in 0.3.6
------------
- Unsupervised (1-class) kernel density modeling
- Fix for when SDA returns empty
- weights option to some learners
- stump learner
- Adaboost (result of above changes)

New in 0.3.5
------------
- Fixes for 64 bit machines
- Functions in measures.py all have same interface now.

New in 0.3.4
------------
- Random forest learners
- Decision trees sped up 20x
- Much faster gridsearch  (finds optimum without computing all folds)

Features
--------
- Random forests
- Self organising maps
- SVMs. Using the libsvm solver with a pythonesque wrapper around it.
- Stepwise Discriminant Analysis for feature selection.
- Non-negative matrix factorisation
- K-means using as little memory as possible.
- Affinity propagation

License: MIT
Author: Luis Pedro Coelho (with code from LibSVM and scikits.learn)
Website: `http://luispedro.org/software/milk
<http://luispedro.org/software/milk>`__
API Documentation: `http://packages.python.org/milk/ <http://packages.python.org/milk/>`_
