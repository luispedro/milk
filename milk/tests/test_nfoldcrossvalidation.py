import milk.measures.nfoldcrossvalidation
from milk.measures.nfoldcrossvalidation import nfoldcrossvalidation, foldgenerator
import milk.supervised.tree
import numpy as np
from fast_classifier import fast_classifier

def test_foldgenerator():
    labels = np.array([1]*20+[2]*30+[3]*20)
    for nf in [None,2,3,5,10,15,20]:
        assert np.array([test.copy() for _,test in foldgenerator(labels,nf)]).sum(0).max() == 1
        assert np.array([test.copy() for _,test in foldgenerator(labels,nf)]).sum() == len(labels)
        assert np.array([(test&train).sum() for train,test in foldgenerator(labels,nf)]).sum() == 0

def test_foldgenerator_not_empty():
    for nf in (None, 2, 3, 5, 10, 15, 20):
        for Tr,Te in foldgenerator([0] * 10 + [1] *10, nf, None):
            assert not np.all(Tr)
            assert not np.all(Te)




def test_nfoldcrossvalidation_simple():
    from milksets import wine
    features, labels = wine.load()
    features = features[::2]
    labels = labels[::2]

    cmat,clabels = nfoldcrossvalidation(features, labels, classifier=fast_classifier())
    assert cmat.shape == (3,3)
    assert len(clabels) == 3

def test_nfoldcrossvalidation_simple_list():
    from milksets import wine
    features, labels = wine.load()
    features = features[::2]
    labels = labels[::2]

    cmat,clabels = nfoldcrossvalidation(list(features), list(labels), classifier=fast_classifier())
    assert cmat.shape == (3,3)
    assert len(clabels) == 3

class test_classifier(object):
    def __init__(self,N):
        self.tested = np.zeros(N,bool)
        self.N = N
    def apply(self, features):
        cur = np.zeros(self.N,bool)
        cur[features] = True
        assert not np.any(cur & self.tested)
        self.tested |= cur
        return np.zeros_like(features)
    def train(self,f,l):
        return self

def test_nfoldcrossvalidation_testall():
    N = 121
    C = test_classifier(N)
    features = np.arange(N)
    labels = np.zeros(N)
    cmat,clabels = nfoldcrossvalidation(features, labels, classifier=C)
    assert np.all(C.tested)

def test_getfold():
    A = np.zeros(20)
    A[:10] = 1
    t,s = milk.measures.nfoldcrossvalidation.getfold(A,0,10)
    tt,ss = milk.measures.nfoldcrossvalidation.getfold(A,1,10)
    assert not np.any((~t)&(~tt))

def test_nfoldcrossvalidation_defaultclassifier():
    np.random.seed(2233)
    X = np.random.rand(60,5)
    X[:20] += 4.
    X[-20:] -= 4.
    Y = np.ones(60)
    Y[:20] = 0
    Y[-20:] = 2
    Y += 100
    cmat,clabels = milk.measures.nfoldcrossvalidation.nfoldcrossvalidation(X,Y)
    assert cmat.shape == (3,3)
    clabels.sort()
    assert np.all(clabels == [100,101,102])


def test_foldgenerator_origins():
    def test_origins(labels, origins):
        for nf in (2,3,5,7):
            assert np.array([test.copy() for _,test in foldgenerator(labels, nf, origins)]).sum(0).max() == 1
            assert np.array([test.copy() for _,test in foldgenerator(labels, nf, origins)]).sum() == len(labels)
            assert np.min([test.sum() for _,test in foldgenerator(labels, nf, origins)]) > 0
            assert np.min([train.sum() for train,_ in foldgenerator(labels, nf, origins)]) > 0
            for Tr,Te in foldgenerator(labels, nf, origins):
                assert not np.any(Tr&Te)
                in_test = set(origins[Te])
                in_train = set(origins[Tr])
                assert len(in_train.intersection(in_test)) == 0
            tested = np.zeros(len(labels))
            for Tr,Te in foldgenerator(labels, nf, origins):
                tested[Te] += 1
            assert np.all(tested == 1)
    labels = np.zeros(120, np.uint8)
    labels[39:] += 1
    labels[66:] += 1
    origins = np.repeat(np.arange(40), 3)
    yield test_origins, labels, origins
    reorder = np.argsort(np.random.rand(len(labels)))

    labels = labels[reorder]
    origins = origins[reorder]
    yield test_origins, labels, origins


def test_stringlabels():
    np.random.seed(222)
    D = np.random.rand(100,10)
    D[:40] += np.random.rand(40,10)**2
    labelnames = ['one'] * 40 + ['two'] * 60
    cmat,Lo = nfoldcrossvalidation(D, labelnames, classifier=fast_classifier())
    assert Lo[0] in labelnames
    assert Lo[1] in labelnames
    assert Lo[0] != Lo[1] in labelnames

def test_predictions():
    np.random.seed(222)
    D = np.random.rand(100,10)
    D[:40] += np.random.rand(40,10)**2
    labels = [0] * 40 + [1] * 60
    cmat,_,predictions = nfoldcrossvalidation(D, labels, classifier=fast_classifier(), return_predictions=1)
    assert np.all((predictions == 0)|(predictions == 1))
    assert cmat.trace() == np.sum(predictions == labels)

def test_multi():
    np.random.seed(30)
    r = np.random.random
    for _ in xrange(10):
        labels = []
        p = np.array([.24,.5,.1,.44])
        for i in xrange(100):
            cur = [j for j in xrange(4) if r() < p[j]]
            if not cur: cur = [0]
            labels.append(cur)


        seen = np.zeros(100, int)
        for Tr,Te in foldgenerator(labels, 5, multi_label=True):
            assert np.sum(Tr & Te) == 0
            seen[Te] += 1
        assert np.sum(seen) == 100
        assert np.ptp(seen) == 0
