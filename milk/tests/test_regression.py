import pickle
import numpy as np
import milk.supervised._svm
from gzip import GzipFile
from os import path
import numpy as np
from milksets.wine import load
from milk.supervised import defaultclassifier
import milk

def test_svm_crash():
    X,Y,kernel, C, eps ,tol, = pickle.load(GzipFile(path.dirname(__file__) + '/data/regression-2-Dec-2009.pp.gz'))
    X = X[2:-2,:].copy()
    Y = Y[2:-2].copy()
    N = len(Y)
    Y = Y.astype(np.int32)
    p = -np.ones(N,np.double)
    params = np.array([0,C,eps,tol],np.double)
    Alphas0 = np.zeros(N, np.double)
    cache_size = (1<<20)
    # The line below crashed milk:
    milk.supervised._svm.eval_LIBSVM(X,Y,Alphas0,p,params,kernel,cache_size)
    # HASN'T CRASHED!


def test_nov2010():
    # Bug submitted by Mao Ziyang
    # This was failing in 0.3.5 because SDA selected no features
    np.random.seed(222)
    features = np.random.randn(100,20)
    features[:50] *= 2
    labels = np.repeat((0,1), 50)

    classifier = milk.defaultclassifier()
    model = classifier.train(features, labels)
    new_label = model.apply(np.random.randn(20)*2)
    new_label2 = model.apply(np.random.randn(20))
    assert new_label == 0
    assert new_label2 == 1

def test_default_small():
    features, labels = load()
    selected = np.concatenate( [np.where(labels < 2)[0], np.where(labels == 2)[0][:6]] )
    features = features[selected]
    labels = labels[selected]
    learner = defaultclassifier('fast')
    # For version 0.3.8, the line below led to an error
    milk.nfoldcrossvalidation(features, labels, classifier=learner)

