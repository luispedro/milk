# -*- coding: utf-8 -*-
# Copyright (C) 2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
import numpy as np
from milk.unsupervised.kmeans import select_best_kmeans, assign_centroids
from .base import supervised_model, base_adaptor
import multiprocessing
from milk.utils import parallel
from milk import defaultlearner

class precluster_model(supervised_model):
    def __init__(self, centroids, base):
        self.centroids = centroids
        self.base = base
        self.normalise = True

    def apply(self, features):
        histogram = assign_centroids(features, self.centroids, histogram=True, normalise=self.normalise)
        return self.base.apply(histogram)


class precluster_learner(object):
    '''
    This learns a classifier by clustering the input features
    '''
    def __init__(self, ks, base=None, R=None):
        if base is None:
            base = defaultlearner()
        self.ks = ks
        self.R = R
        self.base = base
        self.normalise = True

    def set_option(self, k, v):
        if k in ('R', 'ks'):
            setattr(self, k, v)
        else:
            self.base.set_option(k,v)

    def train(self, features, labels, **kwargs):
        allfeatures = np.vstack(features)
        assignments, centroids = select_best_kmeans(allfeatures, self.ks, repeats=1, method="AIC", R=self.R)
        histograms = [assign_centroids(f, centroids, histogram=True, normalise=self.normalise) for f in features]
        base_model = self.base.train(histograms, labels, **kwargs)
        return precluster_model(centroids, base_model)

class codebook_model(supervised_model):
    def __init__(self, centroids, base, normalise):
        self.centroids = centroids
        self.base = base
        self.normalise = normalise

    def apply(self, features):
        from milk.unsupervised.kmeans import assign_centroids
        f0,f1 = features
        features = assign_centroids(f0, self.centroids, histogram=True, normalise=self.normalise)
        if f1 is not None and len(f1):
            features = np.concatenate((features, f1))
        return self.base.apply(features)


class codebook_learner(base_adaptor):
    def set_option(self, k, v):
        assert k == 'codebook'
        self.codebook = v

    def train(self, features, labels, **kwargs):
        from milk.unsupervised.kmeans import assign_centroids
        tfeatures = np.array([ assign_centroids(f, self.codebook, histogram=True, normalise=self.normalise)
                        for f,_ in features])
        tfeatures = np.hstack((tfeatures, np.array([f for _,f in features])))
        base_model = self.base.train(tfeatures, labels, **kwargs)
        return codebook_model(self.codebook, base_model, self.normalise)

class kmeans_cluster(multiprocessing.Process):
    def __init__(self, features, inq, outq):
        self.features = features
        self.inq = inq
        self.outq = outq

    def execute(self):
        import milk
        while True:
            k,ri = self.inq.get()
            if k == 'shutdown':
                return
            _,centroids = milk.kmeans(self.features, k=k, R=(k*1024+ri))
            self.outq.put(centroids)

    def run(self):
        try:
            self.execute()
        except Exception, e:
            errstr = r'''\
Error in milk.supervised.precluster.learn_codebook internal

Exception was: %s

Original Traceback:
%s

(Since this was run on a different process, this is not a real stack trace).
''' % (e, traceback.format_exc())
            self.outq.put( ('error', errstr) )


class select_precluster(object):

    def __init__(self, ks, base, normalise=True):
        self.base = base
        self.ks = ks
        self.rmax = 16
        self.sample = 16
        self.nfolds = 5
        self.normalise = normalise

    def train(self, features, labels, **kwargs):
        from milk.supervised.gridsearch import gridminimise
        c_features = np.concatenate([f for f,_ in features if len(f)])
        c_features = c_features[::self.sample]
        nprocs = parallel.get_procs(use_current=True)
        tow = multiprocessing.Queue()
        fromw = multiprocessing.Queue()
        for k in self.ks:
            for ri in xrange(self.rmax):
                tow.put((k,ri))
        for i in xrange(nprocs):
            tow.put(('shutdown',None))
        workers = [kmeans_cluster(c_features, tow, fromw) for i in xrange(nprocs)]
        for w in workers:
            if nprocs > 1:
                w.start()
            else:
                w.execute()
        try:
            codebooks = [fromw.get() for i in xrange(len(self.ks)*self.rmax)]
        finally:
            tow.close()
            tow.join_thread()
            if nprocs > 1:
                for w in workers:
                    w.join()
            parallel.release_procs(len(workers), count_current=True)

        base = codebook_learner(self.base)
        base.normalise = self.normalise
        if len(codebooks) > 1:
            (best,) = gridminimise(base, features, labels, { 'codebook' : codebooks }, nfolds=self.nfolds)
            _,codebook = best
        else:
            (codebook,) = codebooks
        base.codebook = codebook
        return base.train(features, labels)

