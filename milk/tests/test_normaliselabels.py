from milk.supervised.normalise import normaliselabels
import numpy as np

def test_normaliselabels():
    np.random.seed(22)
    labels = np.zeros(120, np.uint8)
    labels[40:] += 1
    labels[65:] += 1
    reorder = np.argsort(np.random.rand(len(labels)))
    labels = labels[reorder]
    labels2,names = normaliselabels(labels)
    for new_n,old_n in enumerate(names):
        assert np.all( (labels == old_n) == (labels2 == new_n) )

def test_normaliselabels_multi():
    np.random.seed(30)
    r = np.random.random
    for v in xrange(10):
        labels = []
        p = np.array([.24,.5,.1,.44])
        for i in xrange(100):
            cur = [j for j in xrange(4) if r() < p[j]]
            if not cur: cur = [0]
            labels.append(cur)
        nlabels, names = normaliselabels(labels, True)
        assert len(labels) == len(nlabels)
        assert len(nlabels[0]) == max(map(max,labels))+1
        assert nlabels.sum() == sum(map(len,labels))

