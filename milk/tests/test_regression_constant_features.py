import milk
import numpy as np
def test_constant_features():
    learner = milk.defaultclassifier()
    features = np.ones(20).reshape((-1,1))
    labels = np.zeros(20)
    labels[10:] += 1
    features[10:] *= -1
    learner.train(features, labels)

