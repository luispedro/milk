from milk.measures.curves import precision_recall
import numpy as np
def test_precision_recall():
    labels = [0,1]*10
    values = np.linspace(0,1,len(labels))
    precision, recall = precision_recall(values, labels)
    assert np.min(recall) >= 0.
    assert np.max(recall) <= 1.
    assert np.max(precision) <= 1.
    assert np.min(precision) >= 0.
