# -*- coding: utf-8 -*-
# Copyright (C) 2008, Luís Pedro Coelho <lpc@cmu.edu>
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

from __future__ import division
from .classifier import normaliselabels
from collections import deque
import numpy
import numpy as np
import random
from . import _svm

def _svm_size(SVM):
    '''
    N = _svm_size(SVM)

    Nr. of elements in the SVM

    @internal: This is mostly used for testing. see the class svm_raw
    '''
    return len(SVM[2])

def _svm_apply(SVM,q):
    '''
    f_i = _svm_apply(SVM,q)
   
    @internal: This is mostly used for testing
    '''
    N=_svm_size(SVM)
    X,Y,Alphas,b,C,kernel=SVM
    res = -b
    for i in xrange(N):
        if Alphas[i] != C:
            res += Y[i]*Alphas[i]*kernel(X[i],q)
    return res

def svm_learn_smo(X,Y,kernel,C,eps=1e-4,tol=1e-2,cache_size=(1<<20)):
    '''
    Learn a svm classifier

    X: data
    Y: labels in SVM format (ie Y[i] in (1,-1))


    This is a very raw interface. In general, you should use a class
        like svm_classifier.

    Implements the Sequential Minimum Optimisation Algorithm from Platt's
        "Fast training of support vector machines using sequential minimal optimization"
        in Advances in kernel methods: support vector learning
             Pages: 185 - 208   
             Year of Publication: 1999 
             ISBN:0-262-19416-3
    '''
    assert numpy.all(numpy.abs(Y) == 1)
    assert len(X) == len(Y)
    N = len(Y)
    Y = Y.astype(numpy.int32)
    params = numpy.array([0,C,1e-3,1e-5],numpy.double)
    Alphas0 = numpy.zeros(N, numpy.double)
    _svm.eval_SMO(X,Y,Alphas0,params,kernel,cache_size)
    return Alphas0, params[0]

def svm_learn_libsvm(X,Y,kernel,C,eps=1e-4,tol=1e-2,cache_size=(1<<20)):
    '''
    Learn a svm classifier using LIBSVM optimiser

    X: data
    Y: labels in SVM format (ie Y[i] in (1,-1))

    This is a very raw interface. In general, you should use a class
        like svm_classifier.

    This uses the LIBSVM optimisation algorithm
    '''
    assert numpy.all(numpy.abs(Y) == 1)
    assert len(X) == len(Y)
    N = len(Y)
    Y = Y.astype(numpy.int32)
    p = -numpy.ones(N,numpy.double)
    params = numpy.array([0,C,eps,tol],numpy.double)
    Alphas0 = numpy.zeros(N, numpy.double)
    _svm.eval_LIBSVM(X,Y,Alphas0,p,params,kernel,cache_size)
    return Alphas0, params[0]


def rbf_kernel(sigma,beta=1):
    '''
    kernel = rbf(sigma,beta=1)

    Radial Basis Function kernel

    Returns a kernel (ie, a function that implements)
        beta * exp( - ||x1 - x2|| / sigma) 
    '''
    def k(x1,x2):
        d2=((x1-x2)**2).sum()
        res = beta*numpy.exp(-d2/sigma)
        return res
    k.kernel_nr_ = 0
    k.kernel_arg_ = float(sigma)
    return k

def polynomial_kernel(d,c=1):
    '''
    kernel = polynomial_kernel(d,c=1)

    returns a kernel (ie, a function) that implements:
        (<x1,x2>+c)**d
    '''
    def k(x1,x2):
        return (numpy.dot(x1,x2)+c)**d
    return k

def precomputed_kernel(kmatrix):
    '''
    kernel = precomputed_kernel(kmatrix)

    A "fake" kernel which is precomputed.
    '''
    kmatrix = np.ascontiguousarray(kmatrix,np.double)
    def k(x1,x2):
        return kmatrix[x1,x2]
    k.kernel_nr_ = 1
    k.kernel_arg_ = 0.
    return k

class svm_raw(object):
    '''
    svm_raw: classifier

    classifier = svm_raw(kernel,C,eps=1e-3,tol=1e-8)

    MAJOR PARAMETERS:
    * kernel: the kernel to use.
        This should be a function that takes two data arguments
        see rbf_kernel and polynomial_kernel.
    * C: the C parameter

    MINOR PARAMETERS
    * eps: the precision to which to solve the problem (default 1e-3)
    * tol: (|x| < tol) is considered zero
    '''
    def __init__(self, kernel=None, C=1., eps=1e-3, tol=1e-8):
        self.C = C
        self.kernel = kernel
        self.eps = eps
        self.tol = tol
        self.trained = False
        self.algorithm = 'libsvm'

    def train(self,features,labels):
        assert self.kernel is not None, 'milk.supervised.svm_raw.train: kernel not set!'
        assert self.algorithm in ('libsvm','smo'), 'milk.supervised.svm_raw: unknown algorithm (%s)' % self.algorithm
        self.Y,_ = normaliselabels(labels)
        assert numpy.all( (self.Y == 0) | (self.Y  == 1) ), 'milk.supervised.svm_raw can only handle binary problems'
        self.Y *= 2
        self.Y -= 1
        kernel = self.kernel
        try:
            kernel = (self.kernel.kernel_nr_, self.kernel.kernel_arg_)
            features = np.ascontiguousarray(features, np.double)
        except:
            pass
        if self.algorithm == 'smo':
            alphas,self.b = svm_learn_smo(features,self.Y,kernel,self.C,self.eps,self.tol)
        else:
            alphas,self.b = svm_learn_libsvm(features,self.Y,kernel,self.C,self.eps,self.tol)
        svs = (alphas != 0) & (alphas != self.C)
        self.svs = features[svs]
        self.w = alphas[svs]
        self.Y = self.Y[svs]
        self.trained = True
    
    def get_params(self):
        return self.C, self.eps,self.tol

    def set_params(self,params):
        self.C,self.eps,self.tol = params

    def set_option(self, optname, value):
        setattr(self, optname, value)

    def apply(self,x):
        assert self.trained
        return _svm_apply((self.svs,self.Y,self.w,self.b,self.C,self.kernel),x)


def learn_sigmoid_constants(F,Y,
            max_iters=None,
            min_step=1e-10,
            sigma=1e-12,
            eps=1e-5):
    '''
    A,B = learn_sigmoid_constants(F,Y)

    This is a very low-level interface look into the svm_classifier class.

    Major Arguments
    ---------------

    * F: Values of the function F
    * Y: Labels (in boolean format, ie, in (0,1))

    Minor Arguments
    ---------------

    * max_iters: Maximum nr. of iterations
    * min_step:  Minimum step
    * sigma:     sigma
    * eps:       A small number

    REFERENCE
    ---------
    Implements the algorithm from "A Note on Platt's Probabilistic Outputs for
    Support Vector Machines" by Lin, Lin, and Weng.
    Machine Learning, Vol. 68, No. 3. (23 October 2007), pp. 267-276
    '''
    # the deci[i] array is called F[i] in this code
    assert len(F) == len(Y)
    assert numpy.all( (Y == 1) | (Y == 0) )
    from numpy import log, exp
    N=len(F)
    if max_iters is None: max_iters = 1000

    prior1 = Y.sum()
    prior0 = N-prior1

    small_nr = 1e-4

    hi_t = (prior1+1.)/(prior1+2.)
    lo_t = 1./(prior0+2.)

    T = Y*hi_t + (1-Y)*lo_t

    A = 0.
    B = log( (prior0+1.)/(prior1+1.) )
    def target(A,B):
        fval = 0.
        for i in xrange(N):
            fApB = F[i]*A+B
            if fApB >= 0:
                fval += T[i]*fApB+log(1+exp(-fApB))
            else:
                fval += (T[i]-1.)*fApB+log(1+exp(fApB))
        return fval
    fval = target(A,B)
    for iter in xrange(max_iters):
        h11=sigma
        h22=sigma
        h21=0.
        g1=0.
        g2=0.
        for i in xrange(N):
            fApB = F[i]*A+B
            if (fApB >= 0):
                p = exp(-fApB)/(1.+exp(-fApB))
                q = 1./(1.+exp(-fApB))
            else:
                p = 1./(1.+exp(fApB))
                q = exp(fApB)/(1.+exp(fApB))
            d2 = p * q
            h11 += F[i]*F[i]*d2
            h22 += d2
            h21 += F[i]*d2
            d1 = T[i] - p
            g1 += F[i]*d1
            g2 += d1
        if abs(g1) < eps and abs(g2) < eps: # Stopping criteria
            break
        
        det = h11*h22 - h21*h21
        dA = - (h22*g1 - h21*g2)/det
        dB = - (h21*g1 + h11*g2)/det
        gd = g1*dA + g2*dB
        stepsize = 1.

        while stepsize >= min_step:
            newA = A + stepsize*dA
            newB = B + stepsize*dB
            newf = target(newA,newB)
            if newf < fval+eps*stepsize*gd:
                A = newA
                B = newB
                fval = newf
                break
            stepsize /= 2
        if stepsize < min_step:
            print 'Line search fails'
            break
    return A,B

class svm_binary(object):
    def __init__(self):
        pass

    def train(self,features,labels):
        assert len(labels) >= 2, 'Cannot train from a single example'
        c0 = labels[0]
        i = 1
        while labels[i] == c0:
            i += 1
            if i == len(labels):
                i -= 1
                break
        c1 = labels[i]
        self.classes = [c0,c1]

    def apply(self,f):
        return self.classes[f >= 0.]

class svm_sigmoidal_correction(object):
    '''
    svm_sigmoidal_correction : a classifier

    Sigmoidal approximation for obtaining a probability estimate out of the output
    of an SVM.
    '''
    def __init__(self):
        self.max_iters = None
    
    def train(self,features,labels):
        self.A,self.B = learn_sigmoid_constants(features,labels,self.max_iters)

    def get_params(self):
        return self.max_iters

    def set_params(self,params):
        self.max_iters = params

    def apply(self,features):
        return 1./(1.+numpy.exp(features*self.A+self.B))

def sigma_value_fisher(features,labels):
    '''
    f = sigma_value_fisher(features,labels)
    value_s = f(s)

    Computes a function which computes how good the value of sigma
    is for the features. This function should be *minimised* for a 
    good value of sigma.

    Parameters
    -----------
        * features: the feature matrix.

    Reference
    ----------

    Implements the measure in

        "Determination of the spread parameter in the
        Gaussian kernel for classification and regression"
    by Wenjian Wanga, Zongben Xua, Weizhen Luc, and Xiaoyun Zhanga
    '''
    features = np.asanyarray(features)
    xij = np.dot(features,features.T)
    f2 = np.sum(features**2,1)
    d = f2-2*xij
    d = d.T + f2
    N1 = (labels==0).sum()
    N2 = (labels==1).sum()

    C1 = -d[labels == 0][:,labels == 0]
    C2 = -d[labels == 1][:,labels == 1]
    C12 = -d[labels == 0][:,labels == 1]
    C1 = C1.copy()
    C2 = C2.copy()
    C12 = C12.copy()
    def f(sigma):
        sigma = float(sigma)
        N1 = C1.shape[0]
        N2 = C2.shape[0]
        if C12.shape != (N1,N2):
            raise ValueError
        N1 = int(N1)
        N2 = int(N2)
        try:
            from scipy import weave
            from scipy.weave import converters
            res = np.empty(3,np.double)
            code = '''
#line 400 "svm.py"
            double C1v = 0.;
            double C2v = 0.;
            double C12v = 0.;
            for (int i = 0; i != N1; ++i)
                for (int j = 0; j != N1; ++j) 
                    C1v += std::exp(C1(i,j)/sigma);
            for (int i = 0; i != N2; ++i)
                for (int j = 0; j != N2; ++j)
                    C2v += std::exp(C2(i,j)/sigma);
            for (int i = 0; i != N1; ++i)
                for (int j = 0; j != N2; ++j)
                    C12v += std::exp(C12(i,j)/sigma);
            res(0) = C1v/N1;
            res(1) = C2v/N2;
            res(2) = C12v/N1/N2;
            '''
            weave.inline(code, ['C1','C2','C12','N1','N2','res','sigma'],type_converters = converters.blitz)
            C1v = res[0]
            C2v = res[1]
            C12v = res[2]
        except Exception, e:
            C1v = np.sum(np.exp(C1/sigma))/N1
            C2v = np.sum(np.exp(C2/sigma))/N2
            C12v = np.sum(np.exp(C12/sigma))/N1/N2
        return (N1 + N2 - C1v - C2v)/(C1v/N1+C2v/N2 - 2.*C12v)
    return f

class fisher_tuned_rbf_svm(object):
    '''
    F = fisher_tuned_rbf_svm(sigmas, base)

    Returns a wrapper classifier that uses RBF kernels automatically
    tuned using sigma_value_fisher.

    '''
    def __init__(self, sigmas, base):
        self.sigmas = sigmas
        self.base = base
        self.trained = False

    def train(self, features, labels):
        f = sigma_value_fisher(features, labels)
        fs = [f(s) for s in self.sigmas]
        self.sigma = self.sigmas[np.argmin(fs)]
        self.base.set_option('kernel',rbf_kernel(self.sigma))
        self.base.train(features,labels)
        self.trained = True

    def apply(self,features):
        return self.base.apply(features)

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
