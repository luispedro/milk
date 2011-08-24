===================
Parallel Processing
===================

.. versionadded:: 0.3.10
   Jug integration was added in version 0.3.10. Parallel processing was added
   with 0.4.0

There is certain functionality in milk which is *embarassingly parallel* (or
almost so). Therefore, milk has some support for using multiprocessors and
computing clusters.

Jug Integration
---------------

.. versionadded:: 0.3.10
   Jug integration requires `jug <http://luispedro.org/software/jug>`__

Currently, there is support for running n-fold crossvalidation as multiple jug
tasks, which jug can then partition across multiple processors (or computers in
a cluster).

Example
~~~~~~~

::
    from milk.ext.jugparallel import nfoldcrossvalidation

    # For this example, we rely on milksets
    from milksets.wine import load

    # Load the data
    features, labels = load()

    cmatrix = nfoldcrossvalidation(features, labels)


Save this as ``example.py`` and, now, you can run ``jug execute example.py`` to
perform 10-fold cross-validation. Each fold will be its own Task and can be run
independently of the others.

Multiprocessing
---------------

.. versionadded:: 0.4

There are some opportunities for parallel processing which are hard to fit into
the Jug framework (which is limited to coarse grained parallelisation). For
example, choosing the parameters of a learner (e.g., the SVM learner) through
cross-validation, has a high degree of parallelisation, but is hard to fit into
the jug framework without (1) restructuring the code and (2) doing unnecessary
computation.

Therefore, milk can use multiple processes for this operation, using the Python
``multiprocessing`` module.

Currently, by default, *this functionality is disabled.* Change the value of
``milk.utils.parallel.max_procs`` to enable it.

Over time, more functionality will take advantage of multiple cores.

Example
~~~~~~~

This is a simple example, which relies on `milksets
<http://luispedro.org/software/milksets>`__ just for convenience (you could use
any other labeled feature set.

As you can see, you do not have to do anything except call
``milk.utils.parallel.set_max_procs()`` to enable multiprocessing (calling it
without an argument sets the number of processes to the number of CPUs).

::
    import milk

    # Import the parallel module
    from milk.utils import parallel

    # For this example, we rely on milksets
    from milksets.wine import load

    # Use all available processors
    parallel.set_max_procs()

    # Load the data
    features, labels = load()
    learner = milk.defaultlearner()
    model = learn.train(features[::2], labels[::2])
    held_out = map(model.apply, features[1::2])
    print np.mean(labels[1::2] == held_out)


Naturally, you can combine both of these features::

    from milk.ext.jugparallel import nfoldcrossvalidation
    # Import the parallel module
    from milk.utils import parallel

    # For this example, we rely on milksets
    from milksets.wine import load

    # Use all available processors
    parallel.set_max_procs()

    # Load the data
    features, labels = load()

    cmatrix = nfoldcrossvalidation(features, labels)

This is now a jug script which uses all available processors. This is ideal if
you have a cluster of machines with multiple cores per machine. You can run
different folds on different machines and, internally, each fold will use all
the cores on its machine.

Naturally, if you run multiple folds on the same machine, they will end up
fighting for the same cores and you will get no speedup.

