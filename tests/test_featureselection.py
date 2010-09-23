import milk.supervised.featureselection
import numpy as np
def test_sda():
    from milksets import wine
    features, labels = wine.load()
    selected = milk.supervised.featureselection.sda(features,labels)
    for sel in selected:
        assert sel <= features.shape[1]

def test_linear_independent_features():
    np.random.seed(122)
    X3 = np.random.rand(20,3)
    X = np.c_[X3,X3*2+np.random.rand(20,3)/20.,-X3*2+np.random.rand(20,3)/10.]
    X2 = np.c_[X3,X3*2,-X3*3e-3]
    assert len(milk.supervised.featureselection.linear_independent_features(X)) == 9
    assert len(milk.supervised.featureselection.linear_independent_features(X2)) == 3
    assert np.all (np.sort(milk.supervised.featureselection.linear_independent_features(X2) % 3) == np.arange(3))

