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
import numpy

__all__ = ['accuracy','waccuracy']

def accuracy(cmatrix):
    '''
    acc = accuracy(cmatrix)

    Accuracy of confusion matrix
    '''
    cmatrix = numpy.asanyarray(cmatrix)
    return cmatrix.trace()/cmatrix.sum()

def waccuracy(cmatrix):
    '''
    wacc = waccuracy(cmatrix)

    Weighted accuracy of cmatrix
    '''
    cmatrix = numpy.asanyarray(cmatrix)
    return (cmatrix.diagonal() / cmatrix.sum(1)).mean()

## TODO: Implement http://en.wikipedia.org/wiki/Matthews_Correlation_Coefficient

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:

