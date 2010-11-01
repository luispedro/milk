==============================
MILK: MACHINE LEARNING TOOLKIT
==============================
Machine Learning in Python
--------------------------

Milk is a machine learning toolkit in Python. Its focus is on supervised
classification and on enabling medium scale learning (defined as data that
barely fits in main memory).

milk wraps libsvm in a Pythonic way (the models learned have weight arrays that
are accessible from Python directly, the models are pickle()able, you can pass
any Python function as a kernel,....)

It also supports k-means clustering with an implementation that is careful not
to use too much memory (if your dataset fits into memory, milk can cluster it).

It does not have its own file format or in-memory format, which I consider a
feature as it works on numpy arrays directly (or anything that is convertible to
a numpy-array) without forcing you to copy memory around. For SVMs, you can even
just use any datatype if you have your own kernel function.

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
Website: http://luispedro.org/software/milk
API Documentation: `http://packages.python.org/milk/ <http://packages.python.org/milk/>`_
