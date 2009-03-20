import milk.supervised.featureselection
import tests.data.german.german
data = tests.data.german.german.load()
features = data['data']
labels = data['label']
def test_sda():
    selected = milk.supervised.featureselection.sda(features,labels)
    for sel in selected:
        assert sel <= features.shape[1]
