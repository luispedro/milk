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

'''
Milk

Machine learning in Python

Toplevel functions
------------------
- nfoldcrossvalidation: n-fold crossvalidation
- defaultclassifier: get a general purpose classifier
- kmeans: kmeans clustering

Modules
-------
- supervised
- unsupervised
- measures

Example
-------

::

    features = np.random.randn(100,20)
    features[:50] *= 2
    labels = np.repeat((0,1), 50)

    classifier = milk.defaultclassifier()
    model = classifier.train(features, labels)
    new_label = model.apply(np.random.randn(100))
    new_label2 = model.apply(np.random.randn(100)*2)

'''

try:
    from .nfoldcrossvalidation import nfoldcrossvalidation
    from .supervised.defaultclassifier import defaultclassifier
    from .supervised.defaultlearner import defaultlearner
    from .unsupervised.kmeans import kmeans
    from milk_version import __version__
except ImportError, e:
    import sys
    print >>sys.stderr, '''\
Could not import submodules (exact error was: %s).

There are many reasons for this error the most common one is that you have
either not built the packages or have built (using `python setup.py build`) or
installed them (using `python setup.py install`) and then proceeded to test
milk **without changing the current directory**.

Try installing and then changing to another directory before importing milk.
''' % e

__all__ = [
    '__version__',
    'kmeans',
    'defaultclassifier',
    'defaultlearner',
    'nfoldcrossvalidation',
    ]
