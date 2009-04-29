import milk.unsupervised
import numpy as np
def test_nnmf():
    def test3(method):
        X3 = np.random.rand(20,3)
        X = np.c_[  X3,
                    X3*2+np.random.rand(20,3)/20.,
                    -X3*2+np.random.rand(20,3)/10.]
        W,V = method(X,3)
        assert np.sum((np.dot(W,V)-X)**2)/np.sum(X**2) < .5

    yield test3, milk.unsupervised.lee_seung
    yield test3, milk.unsupervised.sparse_nnmf
