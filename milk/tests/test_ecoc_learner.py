import milk.tests.fast_classifier
import milk.supervised.multi
from milksets.yeast import load
import numpy as np

def test_ecoc_learner():
    base = milk.tests.fast_classifier.fast_classifier()
    learner = milk.supervised.multi.ecoc_learner(base)
    features, labels = load()
    nlabels = len(set(labels))
    model = learner.train(features[::2],labels[::2])

    testl = np.array(model.apply_many(features[1::2]))
    assert np.mean(testl == labels[1::2]) > 1./nlabels
    assert testl.min() >= 0
    assert testl.max() < nlabels
