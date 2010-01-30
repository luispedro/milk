Examples
========

Learning a classification model
-------------------------------

Let us solve this very simple problem: you have a bunch of labeled datapoints (each encoded as a set of 10 features) and you want to build a classifier for them.

::

    data = [
            [1., 1.2, 3.03e+3, 2.34e-9, ...],
            [2., 1.3, 5.8e+5, 2.36e-9, ...],
            [0., 1.1, -1.9e+6, 2.33e-9, ...],
            [1., 1.3, -2.21e-4, 2.35e-9, ...],
            ...
            ]
    labels = [
            'positive',
            'positive',
            'negative',
            ...
            ]


