import numpy as np
from milk.supervised import set2binary_array

def test_set2binary_array_len():
    s2f = set2binary_array.set2binary_array()
    inputs = [ np.arange(1,3)*2, np.arange(4)**2, np.arange(6)+2 ]
    labels = [0,0,1]
    model = s2f.train(inputs,labels)
    assert len(model.apply(inputs[0])) == len(model.apply(inputs[1]))
    assert len(model.apply(inputs[0])) == len(model.apply(inputs[2]))
    assert len(model.apply(inputs[0])) == len(model.apply(range(128)))

