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
        pass
    
def test_nfoldcrossvalidation_testall():
    N = 121
    C = test_classifier(N)
    features = np.arange(N)
    labels = np.zeros(N)
    cmat,clabels = nfoldcrossvalidation(features, labels, classifier=C)
    assert np.all(C.tested)
