// Copyright (C) 2010, Luis Pedro Coelho <lpc@cmu.edu>
// License: MIT

#include <cassert>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <list>
#include <memory>
#include <cmath>
#include <vector>
extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}


namespace {

PyObject* eval_set_entropy(PyObject* self, PyObject* args) {
    const char* errmsg = "Arguments were not what was expected for set_entropy.\n"
                        "This is an internal function: Do not call directly unless you know exactly what you're doing.\n";
    PyArrayObject* labels;
    PyArrayObject* counts;
    if (!PyArg_ParseTuple(args, "OO", &labels, &counts)) {
        PyErr_SetString(PyExc_RuntimeError,errmsg);
        return 0;
    }
    if (!PyArray_Check(labels) || PyArray_TYPE(labels) != NPY_INT32 || !PyArray_ISCONTIGUOUS(labels) ||
        !PyArray_Check(counts) || PyArray_TYPE(counts) != NPY_DOUBLE || !PyArray_ISCONTIGUOUS(counts)) {
        PyErr_SetString(PyExc_RuntimeError,errmsg);
        return 0;
    }
    const int* ldata = reinterpret_cast<const int*>(PyArray_DATA(labels));
    double* cdata = reinterpret_cast<double*>(PyArray_DATA(counts));
    const int N = PyArray_DIM(labels,0);
    const int nlabels = PyArray_DIM(counts, 0);

    for (int i = 0; i != nlabels; ++i) cdata[i] = 0.;
    for (int i = 0; i != N; ++i) {
        int value = ldata[i];
        if (value >= nlabels) {
            PyErr_SetString(PyExc_RuntimeError, "value too large. aborting");
            return 0;
        }
        cdata[value] += 1.;
    }
    // Here is the formula we use:
    //
    // H = - \sum px \log(px)
    //   = - \sum (cx/N) \log( cx / N)
    //   = - 1/N \sum { cx [ \log cx - \log N ] }
    //   = - 1/N { (\sum cx \log cx ) - ( \sum cx \log N ) }
    //   = - 1/N { (\sum cx \log cx ) - N \log N }
    //   = ( - 1/N \sum cx \log cx ) + \log N

    double entropy = 0.;
    for (int i = 0; i != nlabels; ++i) {
        double cx = cdata[i];
        if (cx) entropy += cx * std::log(cx);
    }
    entropy /= -N;
    entropy += std::log(N);
    return PyFloat_FromDouble(entropy);
}

PyMethodDef methods[] = {
  {"set_entropy", eval_set_entropy, METH_VARARGS , "Do NOT call directly.\n" },
  {NULL, NULL,0,NULL},
};

const char  * module_doc =
    "Internal Module.\n"
    "\n"
    "Do NOT use directly!\n";

} // namespace

extern "C"
void init_tree()
  {
    import_array();
    (void)Py_InitModule3("_tree", methods, module_doc);
  }

