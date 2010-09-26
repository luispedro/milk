import milk.unsupervised
import numpy as np
def test_nnmf():
    def test3(method):
        np.random.seed(8)
        X3 = np.random.rand(20,3)
        X = np.c_[  X3,
                    X3*2+np.random.rand(20,3)/20.,
                    -X3*2+np.random.rand(20,3)/10.]
        W,V = method(X, 3, R=7)
        assert np.sum((np.dot(W,V)-X)**2)/np.sum(X**2) < .5

    yield test3, milk.unsupervised.lee_seung
    yield test3, milk.unsupervised.sparse_nnmf

def test_sparse_nnmf():
    # This is really just a smoke test because the test case is not sparse!!
    from milk.unsupervised import sparse_nnmf
    np.random.seed(8)
    X3 = np.random.rand(20,3)
    X = np.c_[  X3,
                X3*2+np.random.rand(20,3)/20.,
                -X3*2+np.random.rand(20,3)/10.]
    W,V = sparse_nnmf(X, 3, sparsenessW=.7, sparsenessH=.7, R=7)
    assert not np.any(np.isnan(W))
    assert not np.any(np.isnan(V))
    error = np.dot(W,V)-X
    assert error.var() < X.var()



def test_hoyer_project():
    from milk.unsupervised.nnmf.hoyer import _L1for, _project
    def sp(n, L1, L2):
        return (np.sqrt(n) - L1/L2)/(np.sqrt(n) - 1)
    sparseness = .6
    n = 9.
    row = np.arange(int(n))/n
    L2 = np.sqrt(np.dot(row, row))
    L1 = _L1for(sparseness, row, L2)

    assert np.abs(sp(n, L1, L2) - sparseness) < 1.e-4
    row_ = _project(row, L1, L2)
    assert not np.any(np.isnan(row_))
    assert np.all(row_ >= 0)

    L2 = np.sqrt(np.dot(row, row))
    L1 = np.sum(np.abs(row_))
    res = sp(n, L1, L2)
    assert np.abs(res - sparseness) < 1.e-4

