#include <algorithm>

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace {
struct Kmeans_Exception {
    Kmeans_Exception(const char* msg): msg(msg) { }
    const char* msg;

};
void assert_type_contiguous(PyArrayObject* array,int type) { 
    if (!PyArray_Check(array) ||
        PyArray_TYPE(array) != type ||
        !PyArray_ISCONTIGUOUS(array)) {
        throw Kmeans_Exception("Arguments to kmeans don't conform to expectation. Are you calling this directly? This is an internal function!");
    }
}

template <typename ftype>
int computecentroids(ftype* f, ftype* centroids, PyArrayObject* a_assignments, PyArrayObject* a_counts, const int N, const int Nf, const int k) {

    int zero_counts = 0;
    Py_BEGIN_ALLOW_THREADS
    const npy_int32* assignments = static_cast<npy_int32*>(PyArray_DATA(a_assignments));
    npy_int32* counts = static_cast<npy_int32*>(PyArray_DATA(a_counts));

    std::fill(counts, counts + k, 0);
    std::fill(centroids, centroids + k*Nf, 0.0);

    for (int i = 0; i != N; i++){
        const int c = assignments[i];
        if (c >= k) continue; // throw Kmeans_Exception("wrong value in assignments");
        ftype* ck = centroids + Nf*c;
        for (int j = 0; j != Nf; ++j) {
            *ck++ += *f++;
        }
        ++counts[c];
    }
    for (int i = 0; i != k; ++i) {
        ftype* ck = centroids + Nf*i;
        const ftype c = counts[i];
        if (c == 0) {
            ++zero_counts;
        } else {
            for (int j = 0; j != Nf; ++j) {
                *ck++ /= c;
            }
        }
    }
    Py_END_ALLOW_THREADS
    return zero_counts;
}


PyObject* py_computecentroids(PyObject* self, PyObject* args) {
    try {
        PyArrayObject* fmatrix;
        PyArrayObject* centroids;
        PyArrayObject* assignments;
        PyArrayObject* counts;
        if (!PyArg_ParseTuple(args, "OOOO", &fmatrix, &centroids, &assignments, &counts)) { throw Kmeans_Exception("Wrong number of arguments for computecentroids."); }
        if (!PyArray_Check(fmatrix) || !PyArray_ISCONTIGUOUS(fmatrix)) throw Kmeans_Exception("fmatrix not what was expected.");
        if (!PyArray_Check(centroids) || !PyArray_ISCONTIGUOUS(centroids)) throw Kmeans_Exception("centroids not what was expected.");
        if (!PyArray_Check(counts) || !PyArray_ISCONTIGUOUS(counts)) throw Kmeans_Exception("counts not what was expected.");
        if (!PyArray_Check(assignments) || !PyArray_ISCONTIGUOUS(assignments)) throw Kmeans_Exception("assignments not what was expected.");
        if (PyArray_TYPE(counts) != NPY_INT32) throw Kmeans_Exception("counts should be int32.");
        //if (PyArray_TYPE(assignments) != NPY_INT32) throw Kmeans_Exception("assignments should be int32.");
        if (PyArray_TYPE(fmatrix) != PyArray_TYPE(centroids)) throw Kmeans_Exception("centroids and fmatrix should have same type.");
        if (PyArray_NDIM(fmatrix) != 2) throw Kmeans_Exception("fmatrix should be two dimensional");
        if (PyArray_NDIM(centroids) != 2) throw Kmeans_Exception("centroids should be two dimensional");
        if (PyArray_NDIM(assignments) != 1) throw Kmeans_Exception("assignments should be two dimensional");

        const int N = PyArray_DIM(fmatrix, 0);
        const int Nf = PyArray_DIM(fmatrix, 1);
        const int k = PyArray_DIM(centroids, 0);
        if (PyArray_DIM(centroids, 1) != Nf) throw Kmeans_Exception("centroids has wrong number of features.");
        if (PyArray_DIM(assignments, 0) != N) throw Kmeans_Exception("assignments has wrong size.");
        if (PyArray_DIM(counts, 0) != k) throw Kmeans_Exception("counts has wrong size.");
        switch (PyArray_TYPE(fmatrix)) {
#define TRY_TYPE(code, ftype) \
            case code: \
                if (computecentroids<ftype>( \
                        static_cast<ftype*>(PyArray_DATA(fmatrix)), \
                        static_cast<ftype*>(PyArray_DATA(centroids)), \
                        assignments, \
                        counts, \
                        N, Nf, k)) { \
                        Py_RETURN_TRUE; \
                } \
                Py_RETURN_FALSE;

            TRY_TYPE(NPY_FLOAT, float);
            TRY_TYPE(NPY_DOUBLE, double);
        }
        throw Kmeans_Exception("Cannot handle this type.");
    } catch (const Kmeans_Exception& exc) {
        PyErr_SetString(PyExc_RuntimeError,exc.msg);
        return 0;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,"Some sort of exception in computecentroids.");
        return 0;
    }
}

PyMethodDef methods[] = {
  {"computecentroids", py_computecentroids, METH_VARARGS , "Do NOT call directly.\n" },
  {NULL, NULL,0,NULL},
};

const char  * module_doc = 
    "Internal _kmeans Module.\n"
    "\n"
    "Do NOT use directly!\n";

} // namespace
extern "C"
void init_kmeans()
  {
    import_array();
    (void)Py_InitModule3("_kmeans", methods, module_doc);
  }

