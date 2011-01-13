from milk.supervised import randomforest
from milk.supervised.multi import one_against_one
import milk.nfoldcrossvalidation
import milk.unsupervised

import pylab
from milksets import wine

# Load 'wine' dataset
features, labels = wine.load()
# random forest learner
rf_learner = randomforest.rf_learner()
# rf is a binary learner, so we transform it into a multi-class classifier
learner = one_against_one(rf_learner)

# cross validate with this learner and return predictions on left-out elements
cmat,names, preds = milk.nfoldcrossvalidation(features, labels, classifier=learner, return_predictions=1)

print 'cross-validation accuracy:', cmat.trace()/float(cmat.sum())

# dimensionality reduction for display
x,v = milk.unsupervised.pca(features)
colors = "rgb" # predicted colour
marks = "xo" # whether the prediction was correct
for (y,x),p,r in zip(x[:,:2], preds, labels):
    c = colors[p]
    m = marks[p == r]
    pylab.plot(y,x,c+m)
pylab.show()

