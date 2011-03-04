import milk.supervised.multi_view
import numpy as np
import milk.supervised.svm
from milk.supervised.defaultclassifier import feature_selection_simple

def test_multi_view():
    from milksets.wine import load
    features, labels = load()
    features0 = features[::10]
    features1 = features[1::10]
    features2 = features[2::10]
    labels0 = labels[::10]
    labels1 = labels[1::10]
    labels2 = labels[2::10]

    assert np.all(labels0 == labels1)
    assert np.all(labels1 == labels2)
    labels = labels0
    train_features = zip(features0,features1,features2)
    test_features = zip(features[3::10], features[4::10], features[5::10])
    base = milk.supervised.classifier.ctransforms(
                feature_selection_simple(),
                milk.supervised.svm.svm_raw(C=128, kernel=milk.supervised.svm.rbf_kernel(4.)),
                milk.supervised.svm.svm_sigmoidal_correction()
                )
    classifier = milk.supervised.multi_view.multi_view_classifier([base,base,base])
    model = classifier.train(train_features, labels == 0)
    assert ([model.apply(f) for f in test_features] == (labels == 0)).mean() > .9
