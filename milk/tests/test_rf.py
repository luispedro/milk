from milk.supervised import randomforest
import numpy as np

def test_rf():
    from milksets import wine
    features, labels = wine.load()
    features = features[labels < 2]
    labels = labels[labels < 2]
    learner = randomforest.rf_learner()
    model = learner.train(features[::5], labels[::5])
    test = [model.apply(f) for f in features]
    assert np.mean(labels == test) > .7

