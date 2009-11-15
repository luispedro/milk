import numpy as np
import milk.supervised.defaultclassifier
import pickle
import tests.data.german.german

def test_defaultclassifier():
    data = tests.data.german.german.load()
    features = data['data']
    labels = data['label']
    C = milk.supervised.defaultclassifier()
    model = C.train(features,labels)
    for f in features:
        assert model.apply(f) in (0,1)
test_defaultclassifier.slow = True

def tests_pickle():
    np.random.seed(23232432)
    X = np.random.rand(100,10)
    labels = np.zeros(100)
    X[50:] += .5
    labels[50:] = 1
    classifier = milk.supervised.defaultclassifier()
    model = classifier.train(X, labels)
    s = pickle.dumps(model)
    model = pickle.loads(s)
    test = [model.apply(x) for x in X]
    test = np.array(test)
    assert (test == labels).mean() > .6
