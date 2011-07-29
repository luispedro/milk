import numpy as np
from milk.supervised.perceptron import perceptron_learner
from milk.supervised import _perceptron
from milksets.yeast import load

def test_raw():
    np.random.seed(23)
    data = np.random.random((100,10))
    data[50:] += .5
    labels = np.repeat((0,1), 50)
    weights = np.zeros((11))
    eta = 0.1
    for i in xrange(20):
        _perceptron.perceptron(data, labels, weights, eta)
    errs =  _perceptron.perceptron(data, labels, weights, eta)
    assert errs < 10

def test_wrapper():
    features,labels = load()
    labels = (labels >= 5)

    learner = perceptron_learner()
    model = learner.train(features, labels)
    test = map(model.apply, features)
    assert np.mean(labels != test) < .35
