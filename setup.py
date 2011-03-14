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
execfile('milk/milk_version.py')
long_description = file('README.rst').read()

kmeans_ext = Extension('milk.unsupervised._kmeans', sources = ['milk/unsupervised/_kmeans.cpp'])
svm_ext = Extension('milk.supervised._svm', sources = ['milk/supervised/_svm.cpp'])
som_ext = Extension('milk.unsupervised._som', sources = ['milk/unsupervised/_som.cpp'])
tree_ext = Extension('milk.supervised._tree', sources = ['milk/supervised/_tree.cpp'])
ext_modules = [kmeans_ext, svm_ext, som_ext, tree_ext]

packages = filter(lambda p: p.startswith('milk'), setuptools.find_packages())

setup(name = 'milk',
      version = __version__,
      description = 'Machine Learning Toolkit',
      long_description = long_description,
      author = u'Luis Pedro Coelho',
      author_email = 'lpc@cmu.edu',
      url = 'http://luispedro.org/software/milk',
      license = 'MIT',
      packages = packages,
      ext_modules = ext_modules,
      )


