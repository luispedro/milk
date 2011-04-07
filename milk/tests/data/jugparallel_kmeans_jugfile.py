import milk.ext.jugparallel
from milksets.wine import load
from milk.tests.fast_classifier import fast_classifier
features,labels = load()

clustered = milk.ext.jugparallel.kmeans_select_best(features, ks=(2,8), repeats=2, max_iters=6)

