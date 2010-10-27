import milk.supervised.tree
import milk.supervised._tree
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


def test_set_entropy():
    labels = np.arange(101)%3
    counts = np.zeros(3)
    entropy = milk.supervised._tree.set_entropy(labels, counts)
    slow_counts = np.array([(labels == i).sum() for i in xrange(3)])
    assert np.all(counts == slow_counts)
    px = slow_counts.astype(float)/ slow_counts.sum()
    slow_entropy = - np.sum(px * np.log(px))
    assert np.abs(slow_entropy - entropy) < 1.e-8

