from milk.supervised.nearestneighbour import NNClassifier

def test_nnclassifier():
    labels=[0,1]
    data=[[0.,0.],[1.,1.]]
    C=NNClassifier()
    C.train(data,labels)
    assert C.apply(data[0]) == 0
    assert C.apply(data[1]) == 1
    assert C.apply([.01,.01]) == 0
    assert C.apply([.99,.99]) == 1
    assert C.apply([100,100]) == 1
    assert C.apply([-100,-100]) == 0
    assert C.apply([.9,.9]) == 1
    middle = C.apply([.5,.5])
    assert (middle == 0) or (middle == 1)
