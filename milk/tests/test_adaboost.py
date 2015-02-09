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
    model = learner.train(features[::2], labels[::2])
    train_out = np.array(list(map(model.apply, features)))
    assert (train_out == labels).mean() > .9


def test_too_many_boolean_indices_regression():
    import milk.supervised.randomforest
    import milk.supervised.adaboost
    import milksets.wine
    from milk.supervised.multi import one_against_one

    weak = milk.supervised.randomforest.rf_learner()
    learner = milk.supervised.adaboost.boost_learner(weak)
    learner = one_against_one(learner)

    features, labels = milksets.wine.load()

    # sample features so that the test is faster (still gives error):
    learner.train(features[::16], labels[::16])
