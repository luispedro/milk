# -*- coding: utf-8 -*-
# Copyright (C) 2008-2012, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# License: MIT. See COPYING.MIT file in the milk distribution

from __future__ import division
from .classifier import normaliselabels, ctransforms_model
from collections import deque
import numpy
import numpy as np
import random
from . import _svm

__all__ = [
    'rbf_kernel',
    'polynomial_kernel',
    'precomputed_kernel',
    'dot_kernel',
    'svm_raw',
    'svm_binary',
    'svm_to_binary',
    'svm_sigmoidal_correction',
    'sigma_value_fisher',
    'fisher_tuned_rbf_svm',
    ]


def _svm_apply(SVM, q):
    '''
    f_i = _svm_apply(SVM, q)

    @internal: This is mostly used for testing
    '''
    X,Y,Alphas,b,C,kernel=SVM
    N = len(X)
    s = 0.0
    for i in xrange(N):
        s += Alphas[i] * Y[i] * kernel(q, X[i])
    return s - b

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

def svm_learn_libsvm(features, labels, kernel, C, eps=1e-4, tol=1e-2, cache_size=(1<<20), alphas=None):
    '''
    Learn a svm classifier using LIBSVM optimiser

    This is a very raw interface. In general, you should use a class
        like svm_classifier.

    This uses the LIBSVM optimisation algorithm

    Parameters
    ----------
    X : ndarray
        data
    Y : ndarray
        labels in SVM format (ie Y[i] in (1,-1))
    kernel : kernel
    C : float
    eps : float, optional
    tol : float, optional
    cache_size : int, optional
    alphas : ndarray, optional

    Returns
    -------
    alphas : ndarray
    b : float
    '''
    if not np.all(np.abs(labels) == 1):
        raise ValueError('milk.supervised.svm.svm_learn_libsvm: Y[i] != (-1,+1)')
    assert len(features) == len(labels)
    n = len(labels)
    labels = labels.astype(np.int32)
    p = -np.ones(n, np.double)
    params = np.array([0,C,eps,tol], dtype=np.double)
    if alphas is None:
        alphas = np.zeros(n, np.double)
    elif alphas.dtype != np.double or len(alphas) != n:
        raise ValueError('milk.supervised.svm_learn_libsvm: alphas is in wrong format')
    _svm.eval_LIBSVM(features, labels, alphas, p, params, kernel, cache_size)
    return alphas, params[0]


class preprocessed_rbf_kernel(object):
    def __init__(self, X, sigma, beta):
        self.X = X
        self.Xsum = (X**2).sum(1)
        self.sigma = sigma
        self.beta = beta

    def __call__(self, q):
        minus_d2_sigma = np.dot(self.X,q)
        minus_d2_sigma *= 2.
        minus_d2_sigma -= self.Xsum
        minus_d2_sigma -= np.dot(q,q)
        minus_d2_sigma /= self.sigma
        return self.beta * np.exp(minus_d2_sigma)

class rbf_kernel(object):

    '''
    kernel = rbf_kernel(sigma,beta=1)

    Radial Basis Function kernel

    Returns a kernel (ie, a function that implements)
        beta * exp( - ||x1 - x2|| / sigma)
    '''
    def __init__(self, sigma, beta=1):
        self.sigma = sigma
        self.beta = beta
        self.kernel_nr_ = 0
        self.kernel_arg_ = float(sigma)

    def __call__(self, x1, x2):
        d2 = x1 - x2
        d2 **= 2
        d2 = d2.sum()
        res = self.beta*np.exp(-d2/self.sigma)
        return res

    def preprocess(self, X):
        return preprocessed_rbf_kernel(X, self.sigma, self.beta)

class polynomial_kernel(object):
    '''
    kernel = polynomial_kernel(d,c=1)

    returns a kernel (ie, a function) that implements:
        (<x1,x2>+c)**d
    '''
    def __init__(self, d, c=1):
        self.d = d
        self.c = c

    def __call__(self,x1,x2):
        return (np.dot(x1,x2)+self.c)**self.d

class precomputed_kernel(object):
    '''
    kernel = precomputed_kernel(kmatrix)

    A "fake" kernel which is precomputed.
    '''
    def __init__(self, kmatrix, copy=False):
        kmatrix = np.ascontiguousarray(kmatrix, np.double, copy=copy)
        self.kernel_nr_ = 1
        self.kernel_arg_ = 0.

    def __call__(self, x0, x1):
        return kmatrix[x0,x1]

class _call_kernel(object):
    def __init__(self, k, svs):
        self.svs = svs
        self.kernel = k

    def __call__(self, q):
        return np.array([self.kernel(s, q) for s in self.svs])

class preprocessed_dot_kernel(object):
    def __init__(self, svs):
        self.svs = svs

    def __call__(self, x1):
        return np.dot(self.svs, x1)

class dot_kernel(object):
    def __init__(self):
        self.kernel_nr_ = 2
        self.kernel_arg_ = 0.

    def __call__(self, x0, x1):
        return np.dot(x0, x1)

    def preprocess(self, svs):
        return preprocessed_dot_kernel(svs)

class svm_raw_model(object):
    def __init__(self, svs, Yw, b, kernel):
        self.svs = svs
        self.Yw = Yw
        self.b = b
        self.kernel = kernel
        try:
            self.kernelfunction = self.kernel.preprocess(self.svs)
        except AttributeError:
            self.kernelfunction = _call_kernel(self.kernel, self.svs)

    def apply(self, q):
        Q = self.kernelfunction(q)
        return np.dot(Q, self.Yw) - self.b


class svm_raw(object):
    '''
    svm_raw: classifier

    classifier = svm_raw(kernel, C, eps=1e-3, tol=1e-8)

    Parameters
    ----------
    kernel : the kernel to use.
             This should be a function that takes two data arguments
             see rbf_kernel and polynomial_kernel.
    C : the C parameter

    Other Parameters
    ----------------
    eps : the precision to which to solve the problem (default 1e-3)
    tol : (|x| < tol) is considered zero
    '''
    def __init__(self, kernel=None, C=1., eps=1e-3, tol=1e-8):
        self.C = C
        self.kernel = kernel
        self.eps = eps
        self.tol = tol
        self.algorithm = 'libsvm'


    def train(self, features, labels, normalisedlabels=False, **kwargs):
        assert self.kernel is not None, 'milk.supervised.svm_raw.train: kernel not set!'
        assert self.algorithm in ('libsvm','smo'), 'milk.supervised.svm_raw: unknown algorithm (%s)' % self.algorithm
        assert not (np.isinf(self.C) or np.isnan(self.C)), 'milk.supervised.svm_raw: setting C to NaN or Inf causes problems.'
        features = np.asanyarray(features)
        if normalisedlabels:
            Y = labels.copy()
        else:
            Y,_ = normaliselabels(labels)
        assert Y.max() == 1, 'milk.supervised.svm_raw can only handle binary problems'
        Y *= 2
        Y -= 1
        kernel = self.kernel
        try:
            kernel = (self.kernel.kernel_nr_, self.kernel.kernel_arg_)
            features = np.ascontiguousarray(features, np.double)
        except AttributeError:
            pass
        if self.algorithm == 'smo':
            alphas,b = svm_learn_smo(features,Y,kernel,self.C,self.eps,self.tol)
        else:
            alphas,b = svm_learn_libsvm(features,Y,kernel,self.C,self.eps,self.tol)
        svsi = (alphas != 0)
        svs = features[svsi]
        w = alphas[svsi]
        Y = Y[svsi]
        Yw = w * Y
        return svm_raw_model(svs, Yw, b, self.kernel)

    def get_params(self):
        return self.C, self.eps,self.tol

    def set_params(self,params):
        self.C,self.eps,self.tol = params

    def set_option(self, optname, value):
        setattr(self, optname, value)



def learn_sigmoid_constants(F,Y,
            max_iters=None,
            min_step=1e-10,
            sigma=1e-12,
            eps=1e-5):
    '''
    A,B = learn_sigmoid_constants(F,Y)

    This is a very low-level interface look into the svm_classifier class.

    Parameters
    ----------
    F : Values of the function F
    Y : Labels (in boolean format, ie, in (0,1))

    Other Parameters
    ----------------
    max_iters : Maximum nr. of iterations
    min_step :  Minimum step
    sigma :     sigma
    eps :       A small number

    Reference for Implementation
    ----------------------------
    Implements the algorithm from "A Note on Platt's Probabilistic Outputs for
    Support Vector Machines" by Lin, Lin, and Weng.
    Machine Learning, Vol. 68, No. 3. (23 October 2007), pp. 267-276
    '''
    # Below we use safe constructs to avoid using the overflown values, but we
    # must compute them because of the way numpy works.
    errorstate = np.seterr(over='ignore')

    # the deci[i] array is called F in this code
    F = np.asanyarray(F)
    Y = np.asanyarray(Y)
    assert len(F) == len(Y)
    assert numpy.all( (Y == 1) | (Y == 0) )

    if max_iters is None:
        max_iters = 1000

    prior1 = Y.sum()
    prior0 = len(F)-prior1

    small_nr = 1e-4

    hi_t = (prior1+1.)/(prior1+2.)
    lo_t = 1./(prior0+2.)

    T = Y*hi_t + (1-Y)*lo_t

    A = 0.
    B = np.log( (prior0+1.)/(prior1+1.) )
    def target(A,B):
        fApB = F*A + B
        lef = np.log1p(np.exp(fApB))
        lemf = np.log1p(np.exp(-fApB))
        fvals = np.choose(fApB >= 0, ( T*fApB + lemf, (T-1.)*fApB + lef))
        return np.sum(fvals)

    fval = target(A,B)
    for iter in xrange(max_iters):
        fApB = F*A + B
        ef = np.exp(fApB)
        emf = np.exp(-fApB)

        p = np.choose(fApB >= 0, ( emf/(1.+emf), 1./(1.+ef) ))
        q = np.choose(fApB >= 0, ( 1/(1.+emf), ef/(1.+ef) ))
        d2 = p * q
        h11 = np.dot(F*F,d2) + sigma
        h22 = np.sum(d2) + sigma
        h21 = np.dot(F,d2)
        d1 = T - p
        g1 = np.dot(F,d1)
        g2 = np.sum(d1)
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
        else:
            print 'Line search fails'
            break
    np.seterr(**errorstate)
    return A,B

class svm_binary_model(object):
    def __init__(self, classes):
        self.classes = classes
    def apply(self,f):
        return self.classes[f >= 0.]
class svm_binary(object):
    '''
    classifier = svm_binary()

    model = classifier.train(features, labels)
    assert model.apply(f) in labels
    '''

    def train(self, features, labels, normalisedlabels=False, **kwargs):
        if normalisedlabels:
            return svm_binary_model( (0,1) )
        assert len(labels) >= 2, 'Cannot train from a single example'
        names = sorted(set(labels))
        assert len(names) == 2, 'milk.supervised.svm.svm_binary.train: Can only handle two class problems'
        return svm_binary_model(names)

class svm_to_binary(object):
    '''
    svm_to_binary(base_svm)

    A simple wrapper so that

        svm_to_binary(base_svm)

    is a model that takes the base_svm classifier and then binarises its model output.

    NOTE:  This class does the same job as::

        ctransforms(base_svm, svm_binary())
    '''
    def __init__(self, svm_base):
        '''
        binclassifier = svm_to_binary(svm_base)

        a classifier that binarises the output of svm_base.
        '''
        self.base = svm_base

    def train(self, features, labels, **kwargs):
        model = self.base.train(features, labels, **kwargs)
        binary = svm_binary()
        binary_model = binary.train(features, labels, **kwargs)
        return ctransforms_model([model, binary_model])

    def set_option(self, opt, value):
        self.base.set_option(opt, value)



class svm_sigmoidal_correction_model(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def apply(self,features):
        return 1./(1.+numpy.exp(features*self.A+self.B))

class svm_sigmoidal_correction(object):
    '''
    svm_sigmoidal_correction : a classifier

    Sigmoidal approximation for obtaining a probability estimate out of the output
    of an SVM.
    '''
    def __init__(self):
        self.max_iters = None

    def train(self, features, labels, **kwargs):
        A,B = learn_sigmoid_constants(features,labels,self.max_iters)
        return svm_sigmoidal_correction_model(A, B)

    def get_params(self):
        return self.max_iters

    def set_params(self,params):
        self.max_iters = params


def sigma_value_fisher(features,labels):
    '''
    f = sigma_value_fisher(features,labels)
    value_s = f(s)

    Computes a function which computes how good the value of sigma
    is for the features. This function should be *minimised* for a
    good value of sigma.

    Parameters
    -----------
    features : features matrix as 2-ndarray.

    Returns
    -------
    f : a function: float -> float
        this function should be minimised for a good `sigma`

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

    def train(self, features, labels, **kwargs):
        f = sigma_value_fisher(features, labels)
        fs = [f(s) for s in self.sigmas]
        self.sigma = self.sigmas[np.argmin(fs)]
        self.base.set_option('kernel',rbf_kernel(self.sigma))
        return self.base.train(features, labels, **kwargs)

