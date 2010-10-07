==============
Random Numbers
==============
How milk handles random numbers
-------------------------------

Many algorithms (e.g., `kmeans`) require random number initialisation.

In `milk`, all functions that internally use random numbers take an `R`
parameter. If left unspecified (or set to `None`), then it means that the
internal initialisation should be used.

`R` can be specified by an integer, a `random.Random` instance, or a
`numpy.RandomState` instance. If the same `R` is passed twice to the function,
then the results are deterministic.

Functions that use random numbers
---------------------------------

- `kmeans`: for initial cluster choice.
- `repeated_kmeans`: for use in `kmeans` internally.
- `som`: for initial choice of points.
- `nnmf` and `sparse_nnmf`: for initialisation.

``random`` and ``numpy.random``
-------------------------------

There are two randomness mechanisms used internally by `milk`: `random` (the
standard Python package) and `numpy.random`. Setting the seed on just one of
them will not be enough. You need to set *both*. This is in alternative to
using the `R` technique outlined above.

