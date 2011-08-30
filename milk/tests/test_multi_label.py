from milk.tests.fast_classifier import fast_classifier
import milk.supervised.multi_label
import milk
import numpy as np

def test_one_by_one():
    np.random.seed(23)
    r = np.random.random
    ps = np.array([.7,.5,.8,.3,.8])
    learner = milk.supervised.multi_label.one_by_one(fast_classifier())
    universe = range(len(ps))

    for _ in xrange(10):
        labels = []
        features = []
        bases = [np.random.rand(20) for pj in ps]
        for i in xrange(256):
            cur = []
            curf = np.zeros(20,float)
            for j,pj in enumerate(ps):
                if r() < pj:
                    cur.append(j)
                    curf += r()*bases[j]
            if not cur: continue
            labels.append(cur)
            features.append(curf)

        model = learner.train(features, labels)
        predicted = model.apply_many(features)
        matrix = np.zeros((2,2), int)
        for t,p in zip(labels, predicted):
            for ell in universe:
                row = (ell in t)
                col = (ell in p)
                matrix[row,col] += 1
        Tn,Fp = matrix[0]
        Fn,Tp = matrix[1]
        prec = Tp/float(Tp+Fp)
        recall = Tp/float(Tp+Fn)
        F1 = 2*prec*recall/(prec + recall)
        assert F1 > .3
