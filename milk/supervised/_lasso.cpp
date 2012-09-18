// Copyright (C) 2012, Luis Pedro Coelho <luis@luispedro.org>
// License: MIT

#include <iostream>
#include <memory>
#include <cmath>
#include <random>
#include <cassert>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}


namespace {

template <typename T>
int random_int(T& random, const int max) {
    std::uniform_int_distribution<int> dist(0, max - 1);
    return dist(random);
}

inline
float soft(float val, float lam) {
    return copysign(fdim(fabs(val), lam), val);
}

int coordinate_descent(const MatrixXf& X, const MatrixXf& Y, MatrixXf& B, const int max_iter, const float lam, int maxnops=-1, const float eps=1e-15) {
    std::mt19937 r;
    MatrixXf residuals = Y - B*X;
    if (maxnops == -1) {
        maxnops = 2*B.size();
    }
    int nops = 0;
    int i = 0;
    int j = 0;
    for (int it = 0; it != max_iter; ++it) {
        ++j;
        if (j == B.cols()) {
            j = 0;
            ++i;
            if (i == B.rows()) i = 0;
        }
        // Given everything else as fixed, this comes down to a very simple
        // 1-dimensional problem. We remember the current value:
        const float prev = B(i,j);
        float xy = 0.0;
        float x2 = 0.0;
        for (int k = 0; k != Y.cols(); ++k) {
            if (isnan(Y(i,k))) continue;
            x2 += X(j,k)*X(j,k);
            xy += X(j,k)*residuals(i,k);
        }
        const float step = (x2 == 0. ? 0. : (xy/x2));
        const float best = soft(prev + step, lam);
        if (fabs(best - prev) < eps) {
            ++nops;
            if (nops > maxnops) return it;
        } else {
            assert(!isnan(best));
            nops = 0;
            B(i,j) = best;
            // This is slow, but whatever
            residuals = Y - B*X;
        }
    }
    return max_iter;
}

Map<MatrixXf> as_eigen(PyArrayObject* arr) {
    assert(PyArray_EquivTypenums(PyArray_TYPE(arr), NPY_FLOAT32));
    return Map<MatrixXf>(
                    static_cast<float*>(PyArray_DATA(arr)),
                    PyArray_DIM(arr, 0),
                    PyArray_DIM(arr, 1));
}

const char* errmsg = "INTERNAL ERROR";
PyObject* py_lasso(PyObject* self, PyObject* args) {
    PyArrayObject* Y;
    PyArrayObject* X;
    PyArrayObject* B;
    int max_iter;
    float lam;
    float eps;
    if (!PyArg_ParseTuple(args, "OOOiff", &X, &Y, &B, &max_iter, &lam, &eps)) return NULL;
    if (!PyArray_Check(X) || //!PyArray_ISCARRAY_RO(X) ||
        !PyArray_Check(Y) || //!PyArray_ISCARRAY_RO(Y) ||
        !PyArray_Check(B) || //!PyArray_ISCARRAY(B) ||
        !PyArray_EquivTypenums(PyArray_TYPE(X), NPY_FLOAT32) ||
        !PyArray_EquivTypenums(PyArray_TYPE(Y), NPY_FLOAT32) ||
        !PyArray_EquivTypenums(PyArray_TYPE(B), NPY_FLOAT32)) {
        PyErr_SetString(PyExc_RuntimeError,errmsg);
        return 0;
    }
    MatrixXf mX = as_eigen(X);
    MatrixXf mY = as_eigen(Y);
    MatrixXf mB = as_eigen(B);
    max_iter *= mB.size();
    const int iters = coordinate_descent(mX, mY, mB, max_iter, lam, -1, eps);
    float* rB = static_cast<float*>(PyArray_DATA(B));
    for (int y = 0; y != mB.rows(); ++y) {
        for (int x = 0; x != mB.cols(); ++x) {
            *rB++ = mB(y,x);
        }

    }

    return Py_BuildValue("i", iters);
}

PyMethodDef methods[] = {
  {"lasso", py_lasso, METH_VARARGS , "Do NOT call directly.\n" },
  {NULL, NULL,0,NULL},
};

const char  * module_doc =
    "Internal Module.\n"
    "\n"
    "Do NOT use directly!\n";

} // namespace

extern "C"
void init_lasso()
  {
    import_array();
    (void)Py_InitModule3("_lasso", methods, module_doc);
  }

