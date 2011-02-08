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
for y,x,p in zip(features.T[0], features.T[1], predictions):
    plt.plot([y],[x], colors[p]+'o')
plt.show()

