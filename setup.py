# -*- coding: utf-8 -*-
# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
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

_extensions = {
        'milk.unsupervised._kmeans' : ['milk/unsupervised/_kmeans.cpp'],
        'milk.unsupervised._som' : ['milk/unsupervised/_som.cpp'],

        'milk.supervised._svm' : ['milk/supervised/_svm.cpp'],
        'milk.supervised._tree' : ['milk/supervised/_tree.cpp'],
        'milk.supervised._perceptron' : ['milk/supervised/_perceptron.cpp'],
}
ext_modules = [
    Extension(ext, sources=sources) for ext,sources in _extensions.iteritems()
]

packages = filter(lambda p: p.startswith('milk'), setuptools.find_packages())

setup(name = 'milk',
      version = __version__,
      description = 'Machine Learning Toolkit',
      long_description = long_description,
      author = u'Luis Pedro Coelho',
      author_email = 'luis@luispedro.org',
      url = 'http://luispedro.org/software/milk',
      license = 'MIT',
      packages = packages,
      ext_modules = ext_modules,
      test_suite = 'nose.collector',
      )


