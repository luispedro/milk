import numpy 
from milk.supervised import svm
import numpy
import numpy as np

def old_learn_sigmoid_constants(F,Y,
            max_iters=None,
            min_step=1e-10,
            sigma=1e-12,
            eps=1e-5):
    '''
    Old version. Direct C-like implementation
    '''
    # the deci[i] array is called F[i] in this code
    F = np.asanyarray(F)
    Y = np.asanyarray(Y)
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
            break
    return A,B


def test_learn_sigmoid_contants():
    Y = np.repeat((0,1),100)
    np.random.seed(3)
    for i in xrange(10):
        F = np.random.rand(200)-.3
        F[100:] *= -1
        old = old_learn_sigmoid_constants(F,Y)
        new = svm.learn_sigmoid_constants(F,Y)
        assert np.allclose(old, new)

