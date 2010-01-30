#s -*- coding: utf-8 -*-
# Copyright (C) 2008  Murphy Lab
# Carnegie Mellon University
# 
# Written by Luis Pedro Coelho <lpc@cmu.edu>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
#

from __future__ import division
from numpy import array, asanyarray
_svm = None
try:
    from libsvm import svm
    _svm = svm
except:
    pass

__all__=['libsvmClassifier']
class libsvmClassifier(object):

    def __init__(self,probability = False):
        classifier.__init__(self)
        if _svm is None:
            raise RuntimeError('SVM Library not found. Cannot use this classifier.')
        self.param = _svm.svm_parameter(kernel_type = svm.RBF, probability = probability)
        self.output_probability = False
    
    def set_option(self,optname,value):
        setattr(self.param,optname,value)

    def train(self,features,labels):
        problem = _svm.svm_problem(labels,features)
        return libsvmClassifier_model(_svm.svm_model(problem,self.param))

class libsvmClassifier_model(object):
    def __init__(self, model):
        self.model = model

    def apply(self,feats):
        if self.output_probability:
            return self.model.predict_probability(feats)
        return self.model.predict(feats)
    
    def __getstate__(self):
        # This is really really really hacky, but it works
        N=NamedTemporaryFile()
        self.model.save(N.name)
        S=N.read()
        return S,self.output_probability,self._trained,self._labelnames

    def __setstate__(self,state):
        if _svm is None:
            raise RuntimeError('LibSVM Library not found. Cannot use this classifier.')
        S,self.output_probability,self._trained,self._labelnames = state
        N=NamedTemporaryFile()
        N.write(S)
        N.flush()
        self.model = _svm.svm_model(N.name)

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
