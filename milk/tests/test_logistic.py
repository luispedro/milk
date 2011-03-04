import milk.supervised.logistic
import milksets.wine
import numpy as np
def test_better_than_random():
    learner = milk.supervised.logistic.logistic_learner()
    features, labels = milksets.wine.load()
    model = learner.train(features, labels == 0)
    error = np.array([np.abs(model.apply(f)-(l == 0))
                for f,l in zip(features, labels)])
    assert error.mean() < .1
