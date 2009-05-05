import milk.supervised.gridsearch
import milk.supervised.svm
import tests.data.german.german

def test_gridsearch():
    data = tests.data.german.german.load()
    features = data['data']
    labels = data['label']
    G = milk.supervised.gridsearch.gridsearch(milk.supervised.svm.svm_raw(),C=[.01,.1,1.,10.],kernel=[milk.supervised.svm.rbf_kernel(0.1),milk.supervised.svm.rbf_kernel(1.)])
    G.train(features,labels)
    reslabels = [G.apply(f) for f in features]
    assert len(reslabels) == len(features)

test_gridsearch.slow = True
