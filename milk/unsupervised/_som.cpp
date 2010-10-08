#include <limits>
#include <iostream>
#include <cstdlib>

extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}

namespace {
struct SOM_Exception {
    SOM_Exception(const char* msg): msg(msg) { }
    const char* msg;

};
void assert_type_contiguous(PyArrayObject* array,int type) { 
    if (!PyArray_Check(array) ||
        PyArray_TYPE(array) != type ||
        !PyArray_ISCONTIGUOUS(array)) {
        throw SOM_Exception("Arguments to putpoints don't conform to expectation. Are you calling this directly? This is an internal function!");
    }
}

void putpoints(PyArrayObject* grid, PyArrayObject* points, float L, int radius) {
    if (PyArray_NDIM(grid) != 3) throw SOM_Exception("grid should be three dimensional");
    if (PyArray_NDIM(points) != 2) throw SOM_Exception("points should be two dimensional");
    const int rows = PyArray_DIM(grid, 0);
    const int cols = PyArray_DIM(grid, 1);
    const int d = PyArray_DIM(grid, 2);
    const int n = PyArray_DIM(points, 0);
    if (PyArray_DIM(points, 1) != d) throw SOM_Exception("second dimension of points is not third dimension of grid");

    Py_BEGIN_ALLOW_THREADS

    for (int i = 0; i != n; i++){
        const float* p = static_cast<float*>(PyArray_GETPTR1(points, i));
        int min_y = 0;
        int min_x = 0;
        float best = std::numeric_limits<float>::max();
        for (int y = 0; y != rows; ++y) {
            for (int x = 0; x != cols; ++x) {
                float dist = 0.;
                const float* gpoint = static_cast<float*>(PyArray_GETPTR2(grid, y, x));
                for (int j = 0; j != d; ++j) {
                    dist += (p[j] - gpoint[j])*(p[j] - gpoint[j]);
                }
                if (dist < best) {
                    best = dist;
                    min_y = y;
                    min_x = x;
                }
            }
        }
        const int start_y = std::max(0, min_y - radius);
        const int start_x = std::max(0, min_x - radius);
        const int end_y = std::min(rows, min_y + radius);
        const int end_x = std::min(rows, min_x + radius);
        
        for (int y = start_y; y != end_y; ++y) {
            for (int x = start_x; x != end_x; ++x) {
                const float L2 = L /(1 + std::abs(min_y - y) + std::abs(min_x - x));
                float* gpoint = static_cast<float*>(PyArray_GETPTR2(grid, y, x));
                for (int j = 0; j != d; ++j) {
                    gpoint[j] *= (1.-L2);
                    gpoint[j] += L2 * p[j];
                }
            }
        }
    }
    Py_END_ALLOW_THREADS
}


PyObject* py_putpoints(PyObject* self, PyObject* args) {
    try {
        PyArrayObject* grid; 
        PyArrayObject* points;
        float L;
        int radius;
        if (!PyArg_ParseTuple(args, "OOfi", &grid, &points, &L, &radius)) {
            const char* errmsg = "Arguments were not what was expected for putpoints.\n" 
                                "This is an internal function: Do not call directly unless you know exactly what you're doing.\n";
            PyErr_SetString(PyExc_RuntimeError,errmsg);
            return 0;
        }
        assert_type_contiguous(grid, NPY_FLOAT);
        assert_type_contiguous(points, NPY_FLOAT);
        putpoints(grid, points, L, radius);

        Py_RETURN_NONE;
    } catch (const SOM_Exception& exc) {
        PyErr_SetString(PyExc_RuntimeError,exc.msg);
        return 0;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,"Some sort of exception in putpoints.");
        return 0;
    }
}

PyMethodDef methods[] = {
  {"putpoints", py_putpoints, METH_VARARGS , "Do NOT call directly.\n" },
  {NULL, NULL,0,NULL},
};

const char  * module_doc = 
    "Internal SOM Module.\n"
    "\n"
    "Do NOT use directly!\n";

} // namespace
extern "C"
void init_som()
  {
    import_array();
    (void)Py_InitModule3("_som", methods, module_doc);
  }

