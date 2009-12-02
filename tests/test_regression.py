import pickle
import numpy as np
import milk.supervised._svm
from gzip import GzipFile
from os import path

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
