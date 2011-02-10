import pylab as plt
import milk.supervised.tree
import milk.supervised.adaboost
from milksets import wine
import milk.supervised.multi

weak = milk.supervised.tree.stump_learner()
learner = milk.supervised.adaboost.boost_learner(weak)
learner = milk.supervised.multi.one_against_one(learner)

features, labels = wine.load()
cmat,names,predictions = milk.nfoldcrossvalidation(features,labels, classifier=learner, return_predictions=True)
colors = "rgb"
codes = "xo"
for y,x,r,p in zip(features.T[0], features.T[1], labels, predictions):
    code = codes[int(r == p)]
    plt.plot([y],[x], colors[p]+code)
plt.show()

