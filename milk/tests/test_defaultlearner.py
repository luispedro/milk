import numpy as np
import milk
def test_extra_arg():
    from milksets.wine import load
    features,labels = load()
    learner = milk.defaultlearner()
    model = learner.train(features[::2],labels[::2], extra_arg=5)
    assert model.apply(features[1]) < 12.


def test_empty_input():
    learn = milk.defaultlearner()
    X = np.random.rand(60, 3)
    X[:32] += .52
    y = np.arange(60) > 35
    model = learn.train(X, y)
    preds = model.apply_many([])
    assert len(preds) == 0
