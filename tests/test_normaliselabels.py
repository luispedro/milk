from milk.supervised.classifier import normaliselabels
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

