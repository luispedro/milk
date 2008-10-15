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
from numpy import vectorize
import numpy
import random

def randomize(L):
    L = list(L)
    random.shuffle(L)
    return L

def svm_size(SVM):
    return len(SVM[2])

def svm_apply(SVM,q):
    N=svm_size(SVM)
    X,Y,Alphas,b,C,kernel=SVM
    res = -b
    for i in xrange(N):
        if Alphas[i] != C:
            res += Y[i]*Alphas[i]*kernel(X[i],q)
    return res

def svm_learn(X,Y,kernel,C,eps=1e-3,tol=1e-8):
    assert numpy.all(numpy.abs(Y) == 1)
    assert len(X) == len(Y)
    def f_at(y):
        sum = -thresh[0]
        for i in xrange(N):
            if Alphas[i] != C:
                sum += Y[i]*Alphas[i]*kernel(X[i],y)
        return sum
    def objective_function():
        sum = Alphas.sum()
        for i in xrange(N):
            for j in xrange(N):
                sum -= .5*Y[i]*Y[j]*kernel(X[i],X[j])*Alphas[i]*Alphas[j]
        return sum
        
    def get_error(i):
        if Alphas[i] in (0,C):
            return f_at(X[i])-Y[i]
        return E[i]
    def take_step(i1,i2):
        if i1 == i2: return False
        alpha1 = Alphas[i1]
        alpha2 = Alphas[i2]
        y1 = Y[i1]
        y2 = Y[i2]
        s = y1*y2
        E1 = get_error(i1)
        E2 = get_error(i2)
        if y1 != y2:
            L = max(0,alpha2-alpha1)
            H = min(C,C+alpha2-alpha1)
        else:
            L = max(0,alpha1+alpha2-C)
            H = min(C,alpha1+alpha2)
        #print 'L',L,'H',H
        if L == H:
            return False
        k11 = kernel(X[i1],X[i1])
        k12 = kernel(X[i1],X[i2]) 
        k22 = kernel(X[i2],X[i2]) 
        eta = 2*k12-k11-k22
        if eta < 0:
            a2 = alpha2-y2*(E1-E2)/eta
            a2 = numpy.median((L,a2,H))
        else:
            gamma = alpha1+s*alpha2 # Eq. (12.22)
            v1=E1+y1+thresh[0]-y1*alpha1*k11-y2*alpha2*k12 # Eq. (12.21) # Note that f(x1) = E1 + y1
            v2=E2+y2+thresh[0]-y1*alpha1*k12-y2*alpha2*k22 # Eq. (12.21)
            L_obj = gamma-s*L+L-.5*k11*(gamma-s*L)**2-.5*k22*L**2-s*k12*(gamma-s*L)*L-y1*(gamma-s*L)*v1-y2*L*v2 # + W_const # Eq. (12.23)
            H_obj = gamma-s*H+H-.5*k11*(gamma-s*H)**2-.5*k22*H**2-s*k12*(gamma-s*H)*H-y1*(gamma-s*H)*v1-y2*H*v2 # + W_const # Eq. (12.23)
            if L_obj > H_obj + eps:
                a2 = L
            elif L_obj < H_obj - eps:
                a2 = H
            else:
                a2 = Alphas[i2]
        if a2 < tol:
            a2 = 0
        elif a2 > C-tol:
            a2 = C
        if numpy.abs(a2-alpha2) < eps*(a2+alpha2+eps): return False
        a1 = alpha1+s*(alpha2-a2)
        if a1 < eps: a1 = 0
        if a1 > C-eps: a1 = C

        # update everything
        Alphas[i1]=a1
        Alphas[i2]=a2
        b1 = E1 + Y[i1]*(a1-alpha1)*k11+Y[i2]*(a2-alpha2)*k12+thresh[0] # Eq. (12.9)
        b2 = E2 + Y[i1]*(a1-alpha1)*k12+Y[i2]*(a2-alpha2)*k22+thresh[0] # Eq. (12.10)
        new_b = (b1+b2)/2.
        for i in xrange(N):
            if Alphas[i] in (0,C):
                continue
            elif i == i1 or i == i2:
                E[i] = 0
            else:
                E[i] += y1*(a1-alpha1)*kernel(X[i1],X[i])+y2*(a2-alpha2)*kernel(X[i2],X[i]) + (thresh[0]-new_b) # Eq. (12.11)
        thresh[0] = new_b
        E[i1]=f_at(X[i1])-y1
        E[i2]=f_at(X[i2])-y2
        return True
    def examine_example(i2):
        y2 = Y[i2]
        alpha2 = Alphas[i2]
        E2 = get_error(i2)
        r2 = E2 * y2
        #print 'alpha2', alpha2, 'E2', E2, 'r2', r2
        if (r2 < -tol) and (alpha2 < C) or (r2 > tol) and (alpha2 > 0):
            if Alphas.any() or (Alphas==C).any():
                 dE = numpy.array([numpy.abs(get_error(i)-E2) for i in xrange(N)])
                 dE = numpy.array([numpy.abs(f_at(X[i])-Y[i]) for i in xrange(N)])
                 i1 = dE.argmax()
                 if take_step(i1,i2):
                     return True
            for i1 in randomize(xrange(len(Alphas))):
                if Alphas[i1] and Alphas[i1] != C and take_step(i1,i2):
                    return True
            for i1 in randomize(xrange(len(Alphas))):
                if take_step(i1,i2):
                    return True
        return False
    
    N = len(X)
    Alphas = numpy.zeros(N)
    E = -Y
    # This should be a simple variable.
    # That wouldn't allow one to update it inside the take_step function
    # This is one case where nonlocal would make it work. Til Python3k, we do this.
    thresh = [0.]
    changed = 0
    examineAll = True
    while changed or examineAll:
        changed = 0
        for i in xrange(N):
            if Alphas[i] != 0 or Alphas[i] != C or examineAll:
                changed += examine_example(i)
        if examineAll:
            examineAll = False
        elif not changed:
            examineAll = True
    return Alphas, thresh[0]

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
