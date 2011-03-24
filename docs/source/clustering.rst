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
an effort to make it scale to many millions of data points, the author of milk
even included new features in numpy. If you happen to run numpy 1.6 or newer,
then milk will pick it up and run faster with less memory.


