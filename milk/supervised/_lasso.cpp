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
typedef Map<MatrixXf, Aligned> MapXAf;
struct lasso_solver {
    lasso_solver(const MapXAf& X, const MapXAf& Y, const MapXAf& W, MapXAf& B, const int max_iter, const float lam, int maxnops=-1, const float eps=1e-15)
        :X(X)
        ,Y(Y)
        ,W(W)
        ,B(B)
        ,max_iter(max_iter)
        ,maxnops(maxnops == -1 ? 2*B.size() : maxnops)
        ,lam(lam)
        ,eps(eps)
        { }

    void next_coords(int& i, int& j) {
        ++j;
        if (j == B.cols()) {
            j = 0;
            ++i;
            if (i == B.rows()) i = 0;
        }
    }



    int solve() {
        MatrixXf residuals;
        int nops = 0;
        int i = 0;
        int j = -1;
        for (int it = 0; it != max_iter; ++it) {
            // This resets the residuals matrix to the real values to avoid drift due to approximations
            if (!(it % 512)) residuals = Y - B*X;
            this->next_coords(i, j);

            // We now set βᵢⱼ holding everything else fixed.  This comes down
            // to a very simple 1-dimensional problem.
            // We remember the current value in order to compute update below
            const float prev = B(i,j);
            float xy = 0.0;
            float x2 = 0.0;
            for (int k = 0; k != Y.cols(); ++k) {
                if (std::isnan(Y(i,k))) continue;
                x2 += X(j,k)*X(j,k);
                xy += X(j,k)*residuals(i,k);
                x2 += W(i,k)*X(j,k)*X(j,k);
                xy += W(i,k)*X(j,k)*residuals(i,k);
            }
            const float raw_step = (x2 == 0. ? 0. : (xy/x2));
            const float best = soft(prev + raw_step, lam);
            if (fabs(best - prev) < eps) {
                ++nops;
                if (nops > maxnops) return it;
            } else {
                const float step = best - prev;
                assert(!std::isnan(best));
                nops = 0;
                B(i,j) = best;
                residuals.row(i) -= step*X.row(j);
            }
        }
        return max_iter;
    }
    std::mt19937 r;
    const MapXAf& X;
    const MapXAf& Y;
    const MapXAf& W;
    MapXAf& B;
    const int max_iter;
    const int maxnops;
    const float lam;
    const float eps;
};


MapXAf as_eigen(PyArrayObject* arr) {
    assert(PyArray_EquivTypenums(PyArray_TYPE(arr), NPY_FLOAT32));
    return MapXAf(
                static_cast<float*>(PyArray_DATA(arr)),
                PyArray_DIM(arr, 0),
                PyArray_DIM(arr, 1));
}

const char* errmsg = "INTERNAL ERROR";
PyObject* py_lasso(PyObject* self, PyObject* args) {
    PyArrayObject* X;
    PyArrayObject* Y;
    PyArrayObject* W;
    PyArrayObject* B;
    int max_iter;
    float lam;
    float eps;
    if (!PyArg_ParseTuple(args, "OOOOiff", &X, &Y, &W, &B, &max_iter, &lam, &eps)) return NULL;
    if (!PyArray_Check(X) || //!PyArray_ISCARRAY_RO(X) ||
        !PyArray_Check(Y) || //!PyArray_ISCARRAY_RO(Y) ||
        !PyArray_Check(B) || //!PyArray_ISCARRAY(B) ||
        !PyArray_EquivTypenums(PyArray_TYPE(X), NPY_FLOAT32) ||
        !PyArray_EquivTypenums(PyArray_TYPE(Y), NPY_FLOAT32) ||
        !PyArray_EquivTypenums(PyArray_TYPE(B), NPY_FLOAT32)) {
        PyErr_SetString(PyExc_RuntimeError,errmsg);
        return 0;
    }
    MapXAf mX = as_eigen(X);
    MapXAf mY = as_eigen(Y);
    MapXAf mW = as_eigen(W);
    MapXAf mB = as_eigen(B);
    max_iter *= mB.size();
    lasso_solver solver(mX, mY, mW, mB, max_iter, lam, -1, eps);
    const int iters = solver.solve();

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

