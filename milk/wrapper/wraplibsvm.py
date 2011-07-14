# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
from milk.supervised.classifier import normaliselabels
try:
    from libsvm import svm as libsvm
except ImportError:
    try:
        import svm as libsvm
    except ImportError:
        libsvm = None
from tempfile import NamedTemporaryFile

class libsvmModel(object):
    def __init__(self, model, names, output_probability):
        self.model = model
        self.names = names
        self.output_probability = output_probability

    def apply(self,feats):
        if self.output_probability:
            return self.model.predict_probability(feats)
        res = self.model.predict(feats)
        return self.names[int(res)]

    def __getstate__(self):
        # This is really really really hacky, but it works
        N = NamedTemporaryFile()
        self.model.save(N.name)
        S = N.read()
        return S,self.output_probability,self.names

    def __setstate__(self,state):
        if libsvm is None:
            raise RuntimeError('LibSVM Library not found. Cannot use this classifier.')
        S,self.output_probability,self.names = state
        N = NamedTemporaryFile()
        N.write(S)
        N.flush()
        self.model = libsvm.svm_model(N.name)


class libsvmClassifier(object):
    def __init__(self,probability = False, auto_weighting = True):
        if libsvm is None:
            raise RuntimeError('LibSVM Library not found. Cannot use this classifier.')
        self.param = libsvm.svm_parameter(kernel_type = libsvm.RBF, probability = probability)
        self.output_probability = probability
        self.auto_weighting = auto_weighting

    def set_option(self,optname,value):
        setattr(self.param, optname, value)

    def train(self, features, labels):
        labels,names = normaliselabels(labels)
        if self.auto_weighting:
            nlabels = labels.max() + 1
            self.param.nr_weight = int(nlabels)
            self.param.weight_label = range(nlabels)
            self.param.weight = [(labels != i).mean() for i in xrange(nlabels)]
        problem = libsvm.svm_problem(labels.astype(float), features)
        model = libsvm.svm_model(problem, self.param)
        return libsvmModel(model, names, self.output_probability)

