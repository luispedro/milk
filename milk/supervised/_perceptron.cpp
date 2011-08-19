// Copyright (C) 2011, Luis Pedro Coelho <luis@luispedro.org>
// License: MIT

#include <iostream>
#include <memory>
#include <cmath>
#include <cassert>
extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}


namespace {

template <typename T>
int perceptron(PyArrayObject* data_arr, const long* labels, PyArrayObject* weights_arr, double eta) {
    const T* data = reinterpret_cast<T*>(PyArray_DATA(data_arr));
    T* weights = reinterpret_cast<T*>(PyArray_DATA(weights_arr));
    const int N0 = PyArray_DIM(data_arr, 0);
    const int N1 = PyArray_DIM(data_arr, 1);
    int nr_errors = 0;
    for (int i = 0; i != N0; ++i, data += N1, ++labels) {
        T val = weights[0];
        for (int j = 0; j != N1; ++j) {
            val += weights[j+1] * data[j];
        }
        int ell = (val > 0);
        if (ell != *labels) {
            int pm = (*labels ? +1 : -1);
            ++nr_errors;
            T error = pm * eta * std::abs(pm-val);
            weights[0] += error;
            for (int j = 0; j != N1; ++j) {
                weights[j+1] += error*data[j];
            }
        }
    }
    return nr_errors;
}

PyObject* py_perceptron(PyObject* self, PyObject* args) {
    const char* errmsg = "Arguments were not what was expected for perceptron.\n"
                        "This is an internal function: Do not call directly unless you know exactly what you're doing.\n";
    PyArrayObject* data;
    PyArrayObject* labels;
    PyArrayObject* weights;
    double eta;
    if (!PyArg_ParseTuple(args, "OOOd", &data, &labels, &weights, &eta)) {
        PyErr_SetString(PyExc_RuntimeError,errmsg);
        return 0;
    }
    if (!PyArray_Check(data) || !PyArray_ISCONTIGUOUS(data) ||
        !PyArray_Check(weights) || !PyArray_ISCONTIGUOUS(weights) ||
        !PyArray_Check(labels) || !PyArray_ISCONTIGUOUS(labels) || !PyArray_EquivTypenums(PyArray_TYPE(labels), NPY_LONG) ||
        PyArray_TYPE(data) != PyArray_TYPE(weights)||
        PyArray_NDIM(data) != 2 || PyArray_NDIM(weights) != 1 || PyArray_DIM(data,1) + 1 != PyArray_DIM(weights,0)) {
        PyErr_SetString(PyExc_RuntimeError,errmsg);
        return 0;
    }
    int nr_errors;
    if (PyArray_TYPE(data) == NPY_FLOAT) {
        nr_errors = perceptron<float>(data, reinterpret_cast<const long*>(PyArray_DATA(labels)), weights, eta);
    } else if (PyArray_TYPE(data) == NPY_DOUBLE) {
        nr_errors = perceptron<double>(data, reinterpret_cast<const long*>(PyArray_DATA(labels)), weights, eta);
    } else {
        PyErr_SetString(PyExc_RuntimeError, errmsg);
        return 0;
    }
    return PyLong_FromLong(nr_errors);
}

PyMethodDef methods[] = {
  {"perceptron", py_perceptron, METH_VARARGS , "Do NOT call directly.\n" },
  {NULL, NULL,0,NULL},
};

const char  * module_doc =
    "Internal Module.\n"
    "\n"
    "Do NOT use directly!\n";

} // namespace

extern "C"
void init_perceptron()
  {
    import_array();
    (void)Py_InitModule3("_perceptron", methods, module_doc);
  }

