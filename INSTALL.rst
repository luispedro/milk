=============
Building milk
=============

To install dependencies in Ubuntu::

    sudo apt-get install python-numpy python-scipy libeigen3-dev

The following should work::

    python setup.py install

A C++ compiler is required. On Windows, you might need to specify the compiler.
For example::

    python setup.py install --compiler=mingw32

If you have mingw installed.

