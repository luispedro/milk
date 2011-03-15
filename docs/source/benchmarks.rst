==========
Benchmarks
==========

Scikits.learn benchmark
-----------------------

This is from a benchmark developed by the `scikits.learn team
<https://github.com/scikit-learn/ml-benchmarks>`__. I ran it on my Intel Core2
6600, 2.40GHz CPU.

.. table:: Results in scikits.learn ml-benchmarks

     ============         =======           ======          =======         ========    =============         ======== 
        Benchmark          PyMVPA           Shogun          Pybrain             MLPy    scikits.learn             milk
     ============         =======           ======          =======         ========    =============         ======== 
             knn          **1.0**             2.23               --             2.23             3.05             2.20
      elasticnet               --               --               --           174.43          **1.0**               --
       lassolars               --               --               --            61.67          **1.0**               --
             pca               --               --               --               --          **1.0**            11.11
          kmeans               --             2.02          7057.02             1.61             6.74          **1.0**
             svm             3.35             1.20               --               --             1.24          **1.0**
     ============         =======           ======          =======         ========    =============         ======== 


All of the results are normalised by the fastest system for each entry (which
is therefore, by definition, 1.0).

So, except for PCA, milk *is pretty fast* and for kmeans and SVM learning it is
the fastest system.

Limitations of This Benchmark
-----------------------------

1. It is very small dataset, so you do not get a feeling of how it scales.
2. It is only one dataset.
3. Since the benchmark came out, I made some changes to milk to make it go
   faster. I hope that other systems do the same, though, so we can have good
   progress.

