import numpy as np
import milk
import milk.supervised.defaultclassifier
import pickle

def test_defaultclassifier():
    from milksets import wine
    features, labels = wine.load()
    C = milk.supervised.defaultclassifier()
    model = C.train(features,labels)
    labelset = set(labels)
    for f in features:
        assert model.apply(f) in labelset
test_defaultclassifier.slow = True

def test_pickle():
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

def test_pickle_learner():
    learner = milk.defaultlearner()
    assert len(pickle.dumps(learner))

def test_expandend():
    np.random.seed(23232432)
    X = np.random.rand(100,10)
    labels = np.zeros(100)
    X[50:] += .5
    labels[50:] = 1
    learners = milk.defaultlearner(expanded=True)
    for learner in learners:
        model = learner.train(X, labels)
        test = [model.apply(x) for x in X]
        test = np.array(test)
        assert set(test) == set(labels)

