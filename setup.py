# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010, Luis Pedro Coelho <lpc@cmu.edu>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
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
try:
    import setuptools
except:
    print '''
setuptools not found.

On linux, the package is often called python-setuptools'''
    from sys import exit
    exit(1)
from numpy.distutils.core import setup, Extension

svm_ext = Extension('milk.supervised._svm', sources = ['milk/supervised/_svm.cpp'])
ext_modules = [svm_ext]

packages = filter(lambda p: p.startswith('milk'), setuptools.find_packages())

long_description = '''\
Milk is a machine learning toolkit in Python. Its focus is on supervised classification.

milk wraps libsvm in a Pythonic way (the models learned have weight arrays that
are accessible from Python directly, the models are pickle()able, you can pass
any Python function as a kernel,....)

It also supports k-means clustering with an implementation that is careful not
to use too much memory (if your dataset fits into memory, milk can cluster it).

It does not have its own file format or in-memory format, which I consider a
feature as it works on numpy arrays directly (or anything that is convertible to
a numpy-array) without forcing you to copy memory around. For SVMs, you can even
just use any datatype if you have your own kernel function.

Features
--------
- SVMs. Using the libsvm solver with a pythonesque wrapper around it.
- Stepwise Discriminant Analysis for feature selection.
- K-means using as little memory as possible.

License: MIT
Author: Luis Pedro Coelho (with code from LibSVM)
Website: `http://luispedro.org/software/milk <http://luispedro.org/software/milk>`_
'''

setup(name = 'milk',
      version = '0.1-beta-0',
      description = 'Machine Learning Toolkit',
      long_description = long_description,
      author = u'Luis Pedro Coelho',
      author_email = 'lpc@cmu.edu',
      url = 'http://luispedro.org/software/milk',
      license = 'MIT',
      packages = packages,
      ext_modules = ext_modules,
      )


