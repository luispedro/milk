extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}
#if PY_MAJOR_VERSION < 3

#define DECLARE_MODULE(name) \
extern "C" \
void init##name () { \
    import_array(); \
    (void)Py_InitModule(#name, methods); \
}

#else

#define DECLARE_MODULE(name) \
namespace { \
    struct PyModuleDef moduledef = { \
        PyModuleDef_HEAD_INIT, \
        #name, \
        NULL, \
        -1, \
        methods, \
        NULL, \
        NULL, \
        NULL, \
        NULL \
    }; \
} \
PyMODINIT_FUNC \
PyInit_##name () { \
    import_array(); \
    PyModule_Create(&moduledef); \
}

#endif

