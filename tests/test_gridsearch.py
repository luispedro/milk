import milk.supervised.gridsearch
import milk.supervised.svm
import milk.supervised.gridsearch
import numpy as np

def test_gridsearch():
    from milksets import wine
    features, labels = wine.load()
    selected = (labels < 2)
    features = features[selected]
    labels = labels[selected]

    G = milk.supervised.gridsearch.gridsearch(
            milk.supervised.svm.svm_raw(),
            params={'C':[.01,.1,1.,10.],
                    'kernel':[milk.supervised.svm.rbf_kernel(0.1),milk.supervised.svm.rbf_kernel(1.)]
            })
    model = G.train(features,labels)
    reslabels = [model.apply(f) for f in features]
    assert len(reslabels) == len(features)
test_gridsearch.slow = True


def test_all_assignements():
    assert len(list(milk.supervised.gridsearch._allassignments({'C': [0,1], 'kernel' : ['a','b','c']}))) == 2 * 3



def test_gridmaximise():
    from milksets.wine import load
    features, labels = load()
    x = milk.supervised.gridsearch.gridmaximise(milk.supervised.svm_simple(kernel=np.dot, C=2.), features[::2], labels[::2] == 0, {'C' : (0.5,) })
    cval, = x
    assert cval == ('C', .5)

