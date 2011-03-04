import numpy as np
import milk.supervised.tree
import milk.supervised.adaboost
def test_learner():
    from milksets import wine
    learner = milk.supervised.adaboost.boost_learner(milk.supervised.tree.stump_learner())
    features, labels = wine.load()
    features = features[labels < 2]
    labels = labels[labels < 2] == 0
    labels = labels.astype(int)
    model = learner.train(features, labels)
    train_out = np.array(map(model.apply, features))
    assert (train_out == labels).mean() > .9
