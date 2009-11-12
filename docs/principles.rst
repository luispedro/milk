Principles of Milk
-------------------

Play Well With Others
~~~~~~~~~~~~~~~~~~~~~

This is the basic principle of milk: it should play well with others. It means that its interfaces should, as much as possible, be flexible.

Be Liberal With What you Accept. Be Conservative With What Your Produce.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Don't be fussy about input parameters, but specified very careful your outputs.

Work Interactively
~~~~~~~~~~~~~~~~~~

This means that building a classifier should look like this::

::

    classifier = milk.default_classifier(data,labels)

and not like this::

::

    classifier = milk.concattransforms(
                milk.chkfinite(),
                milk.to_interval(1,-1),
                milk.pick_best(f=0.10),
                milk.binary_to_multi(mode='1-vs-1',
                    base=milk.supervised.gridsearch(
                        base=milk.svm_binary(base=milk.svm_libsvm()),
                        params={ 
                            'C' : [2**c for c in xrange(-7,4)],
                            'kernel' : [milk.rbf_kernel(2**w) for w in xrange(-4,2)])))
    container = milk.container()
    for col in len(data[0]):
        container.set_column(col,milk.CONTINUOUS)
    container.set_data(data)
    labelcontainer = milk.labelcontainer()
    labelcontainer.set_type(milk.STRING)
    labelcontainer.set_data(labels)

    classifier.train(container,labelcontainer)

This often means that one might have a more complete interface internally and another interface for interactive use on top (see Matplotlib_ for a good example of this).

.. _Matplotlib: http://matplotlib.sourceforge.net/


Don't Impose Yourself
~~~~~~~~~~~~~~~~~~~~~

Don't assume that people are writing their software around your library, which translates into:

    * Don't impose your file format.
    * Don't impose your in-memory data format.

Be Pythonic
~~~~~~~~~~~

In general, be a true Python library (and not just a wrapper around something else). For example:

    * If an SVM classifier takes a kernel as a parameter, then it should accept any 2-argument Python function (in fact, anything that's callable in Python).
    * Objects (like classifiers) should be pickle-able.

You Don't Pay For What You Don't Use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Flexibility should come with the lowest-possible cost. If a cost is unavoidable, it should be paid by those who use the flexibility and not by everybody else.
