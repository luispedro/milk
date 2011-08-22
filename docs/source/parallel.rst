===================
Parallel Processing
===================

.. versionadded:: 0.4
   Parallel processing was added with 0.4.0

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

Multiprocessing
---------------

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
