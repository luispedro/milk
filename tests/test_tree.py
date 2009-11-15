import milk.supervised.tree
import numpy as np
import tests.data.german.german

def test_tree():
    data = tests.data.german.german.load()
    features = data['data']
    labels = data['label']
    C = milk.supervised.tree.tree_classifier()
    model = C.train(features,labels)
    assert (np.array([model.apply(f) for f in features]) == labels).mean() > .5

