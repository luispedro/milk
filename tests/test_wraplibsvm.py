import numpy as np
import milk.wrapper.wraplibsvm
def test_wraplibsvm():
    labels = np.zeros(100)
    labels[50:] = 1
    fs = np.random.rand(100,20)
    fs[:50] += .4
    classif = milk.wrapper.wraplibsvm.libsvmClassifier()
    model = classif.train(fs, labels)
    assert np.all([model.apply(f) for f in fs] == labels)

