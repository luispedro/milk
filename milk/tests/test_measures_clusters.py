import milk.measures.cluster_agreement
import numpy as np
def test_rand_arand_jaccard():
    np.random.seed(33)

    labels = np.repeat(np.arange(4),10)
    clusters = np.repeat(np.arange(4),10)

    a0,b0,c0= milk.measures.cluster_agreement.rand_arand_jaccard(clusters, labels)
    assert a0 == 1.
    assert b0 == 1.

    np.random.shuffle(clusters)
    a1,b1,c1= milk.measures.cluster_agreement.rand_arand_jaccard(clusters, labels)
    assert a1 >= 0.
    assert a1 < 1.
    assert b1 < 1.
    assert b1 >= 0.
    assert c1 < c0

