import tests.data.german.german
import milk.supervised.defaultclassifier
data = tests.data.german.german.load()
features = data['data']
labels = data['label']
def test_defaultclassifier():
    C = milk.supervised.defaultclassifier()
    C.train(features,labels)
    for f in features:
        assert C.apply(f) in (0,1)
test_defaultclassifier.slow = True

