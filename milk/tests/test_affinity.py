import milk.unsupervised.affinity
import numpy as np
def test_affinity():
    np.random.seed(22)
    X = np.random.randn(100,10)
    X[:40] += .4
    S = milk.unsupervised.pdist(X)
    clusters, labels = milk.unsupervised.affinity.affinity_propagation(S)
    assert labels.max()+1 == len(clusters)
    assert len(labels) == len(X)
    assert clusters.max() < len(X)
