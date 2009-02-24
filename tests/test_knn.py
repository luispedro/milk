import numpy as np
import milk.supervised.knn
import numpy

def test_simple():
    X=np.array([
        [0,0,0],   
        [1,1,1],   
        ])         
    Y=np.array([ 1, -1 ])
    kNN = milk.supervised.knn.kNN(1)
    kNN.train(X,Y)
    assert kNN.apply(X[0]) == Y[0]
    assert kNN.apply(X[1]) == Y[1]
    assert kNN.apply([0,0,1]) == Y[0]
    assert kNN.apply([0,1,1]) == Y[1]

