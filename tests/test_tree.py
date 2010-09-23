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

