from pylab import *
import numpy as np

from milksets.wine import load
import milk.supervised
import milk.unsupervised.pca
import milk.supervised.svm

features, labels = load()
features = features[labels < 2]
labels = labels[labels < 2]
features,_ = milk.unsupervised.pca(features)
features = features[:,:2]
learner = milk.supervised.svm.svm_raw(kernel=np.dot, C=12)
model = learner.train(features, labels)
w = np.dot(model.svs.T, model.Yw)
b = model.b
x = np.linspace(-.5, .1, 100)
y = -w[0]/w[1]*x + b/w[1]
plot(features[labels == 1][:,0], features[labels == 1][:,1], 'bx')
plot(features[labels == 0][:,0], features[labels == 0][:,1], 'ro')
plot(x,y)
savefig('svm-demo-points.pdf')

clf()







learner = milk.supervised.svm.svm_raw(kernel=milk.supervised.svm.rbf_kernel(1.), C=12)
model = learner.train(features, labels)
Y, X = (np.mgrid[:101,:101]-50)/12.5
values = [model.apply((y,x)) for y,x in zip(Y.ravel(),X.ravel())]
values = np.array(values).reshape(Y.shape)
sfeatures = features*12.5
sfeatures += 50
plot(sfeatures[labels == 0][:,0], sfeatures[labels == 0][:,1], 'bo')
plot(sfeatures[labels == 1][:,0], sfeatures[labels == 1][:,1], 'ro')
imshow(values.T)
savefig('svm-demo-boundary.pdf')


