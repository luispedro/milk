import milk.ext.jugparallel
from milksets.wine import load
from milk.tests.fast_classifier import fast_classifier
features,labels = load()
classified = milk.ext.jugparallel.nfoldcrossvalidation(features, labels, learner=fast_classifier())
classified_wpred = milk.ext.jugparallel.nfoldcrossvalidation(features, labels, learner=fast_classifier(), return_predictions=True)

