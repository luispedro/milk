import milk.measures.nfoldcrossvalidation
from milk.measures.nfoldcrossvalidation import nfoldcrossvalidation
import tests.data.german.german
import milk.supervised.tree
import numpy as np

def test_foldgenerator():
    labels = np.array([1]*20+[2]*30+[3]*20)
    for nf in [None,2,3,5,10,15,20]:
        assert np.array([test.copy() for _,test in milk.measures.nfoldcrossvalidation.foldgenerator(labels,nf)]).sum(0).max() == 1
        assert np.array([test.copy() for _,test in milk.measures.nfoldcrossvalidation.foldgenerator(labels,nf)]).sum() == len(labels)
        assert np.array([(test&train).sum() for train,test in milk.measures.nfoldcrossvalidation.foldgenerator(labels,nf)]).sum() == 0


def test_nfoldcrossvalidation_simple():
    data = tests.data.german.german.load()
    features = data['data']
    labels = data['label']
    C = milk.supervised.tree.tree_classifier()

    cmat,clabels = nfoldcrossvalidation(features, labels, classifier=C)
    assert cmat.shape == (2,2)
    assert len(clabels) == 2

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
    X = np.random.rand(120,5)
    X[:40] += .6
    X[-40:] -= .6
    Y = np.ones(120)
    Y[:40] = 0
    Y[-40:] = 2
    Y += 100
    cmat,clabels = milk.measures.nfoldcrossvalidation.nfoldcrossvalidation(X,Y)
    assert cmat.shape == (3,3)
    clabels.sort()
    assert np.all(clabels == [100,101,102])
