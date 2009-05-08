import tests.data.german.german
import milk.supervised.defaultclassifier
data = tests.data.german.german.load()
features = data['data']
labels = data['label']
def test_defaultclassifier():
    C = milk.supervised.defaultclassifier.defaultclassifier()
    C.train(features,labels)
    assert C.apply(features[0]) in (0,1)
test_defaultclassifier.slow = True

