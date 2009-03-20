import numpy as np 
import milk.unsupervised

def test_kmeans():
    features = np.r_[np.random.rand(20,3)-.5,.5+np.random.rand(20,3)]
    centroids, _ = milk.unsupervised.kmeans(features,2)
    positions = [0]*20 + [1]*20
    correct = (centroids == positions).sum()
    assert correct >= 38 or correct <= 2
