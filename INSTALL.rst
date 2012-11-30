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

---------------
Building on OSX
---------------

Because the standard library used with OS X doesn't include the C++11 libraries by default, you will need to specify
it in ``setup.py``::

    extra_compile_args=['-std=c++0x', '-stdlib=libc++'],

is what the final line should look like.
