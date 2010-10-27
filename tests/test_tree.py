import milk.supervised.tree
import numpy as np

def test_tree():
    from milksets import wine
    features, labels = wine.load()
    selected = (labels < 2)
    features = features[selected]
    labels = labels[selected]
    C = milk.supervised.tree.tree_classifier()
    model = C.train(features,labels)
    assert (np.array([model.apply(f) for f in features]) == labels).mean() > .5


def test_split_subsample():
    import random
    from milksets import wine
    features, labels = wine.load()

    seen = set()
    for i in xrange(20):
        random.seed(2)
        i,s = milk.supervised.tree._split(features[::10], labels[::10], milk.supervised.tree.information_gain, 2, random)
        seen.add(i)
    assert len(seen) <= 2

