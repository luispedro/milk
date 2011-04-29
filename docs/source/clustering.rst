==========
Clustering
==========

K-Means
-------

K-means is one of the simplest, but often most effective, clustering
algorithms. milk supports k-means through the ``milk.kmeans`` function:

::

    features = np.random.randn(100,20)
    features[:50] *= 2

    k = 2
    cluster_ids, centroids = milk.kmeans(features, k)

The milk implementation is very fast and can handle large amounts of data. In
an effort to make it scale to millions of data points, the author of milk even
included new features in numpy. If you happen to run numpy 1.6 or newer, then
milk will pick it up and run faster with less memory.

Milk has been used to cluster datasets with over 5 million data points and over
100 features per data point. You need enough RAM to handle the data matrix and
the distance matrix (NxK) and a little extra, but milk is very careful not to
allocate any more memory than it needs.


