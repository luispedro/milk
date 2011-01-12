from milk.supervised import randomforest
from milk.supervised.multi import one_against_one
import milk.nfoldcrossvalidation
import milk.unsupervised

import pylab
from milksets import wine


features, labels = wine.load()
rf_learner = randomforest.rf_learner()
learner = one_against_one(rf_learner)
cmat,names, preds = milk.nfoldcrossvalidation(features, labels, classifier=learner, return_predictions=1)

print 'cross-validation accuracy:', cmat.trace()/float(cmat.sum())

x,v = milk.unsupervised.pca(features)
colors = "rgb" # predicted colour
marks = "xo" # whether the prediction was correct
for (y,x),p,r in zip(x[:,:2], preds, labels):
    c = colors[p]
    m = marks[p == r]
    pylab.plot(y,x,c+m)
pylab.show()

