// Copyright (C) 2008, Luis Pedro Coelho <luis@luispedro.org>
// Copyright (c) 2000-2008 Chih-Chung Chang and Chih-Jen Lin (LIBSVM Code)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.

#include <cassert>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <list>
#include <memory>
#include <cmath>
#include <vector>
//#include <debug/vector>
//#include <debug/list>
//using __gnu_debug::vector;
//using __gnu_debug::list;

using std::vector;
using std::list;
extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}


namespace { 
const double INF = HUGE_VAL;
// This is a boost function
// Copied here for convenience.
template <typename Iter>
inline Iter prior(Iter it) {
    --it;
    return it;
}
using std::max;
using std::min;

template <typename T>
T median(T bot, T mid, T hi) {
    if (mid < bot) return bot;
    if (mid > hi) return hi;
    return mid;
}

struct SMO_Exception {
    SMO_Exception(const char* msg): msg(msg) { }
    const char* msg;

};

struct Python_Exception { };
void check_for_interrupts() {
    PyErr_CheckSignals();
    if (PyErr_Occurred()) {
        throw Python_Exception();
    }
}

class KernelComputation {
    public:
        virtual ~KernelComputation() { }
        virtual double do_kernel(int i1, int i2) const = 0;
};

class KernelCache {
    public:
        KernelCache(std::auto_ptr<KernelComputation> computation_, int N, int cache_nr_doubles);
        virtual ~KernelCache();
   
        double* get_kline(int idx, int size = -1);
        double* get_diag();
        double kernel_apply(int i1, int i2) const;
    protected:
        virtual double do_kernel(int i1, int i2) const {
            return computation_->do_kernel(i1,i2);
        }
        const int N_;
    private:
        std::auto_ptr<KernelComputation> computation_;
        double ** cache_;
        double * dcache_;
        int cache_free_;
        list<int> cache_lru;
        vector<list<int>::iterator> cache_iter;
};

KernelCache::KernelCache(std::auto_ptr<KernelComputation> computation, int N, int cache_nr_floats):
    N_(N),
    computation_(computation),
    dcache_(0) {
    cache_ = new double*[N_];
    for (int i = 0; i != N_; ++i) cache_[i] = 0;
    cache_free_ = (cache_nr_floats/N_);
    cache_iter.resize(N_,cache_lru.end());
}

KernelCache::~KernelCache() {
    for (int i = 0; i != N_; ++i) delete [] cache_[i];
    delete [] cache_;
    delete [] dcache_;
}

double* KernelCache::get_kline(int idx, int s) {
    if (s == -1) s = N_;
    assert (s <= N_);
    if (!cache_[idx]) {
        if (!cache_free_) {
            int to_remove = cache_lru.front();
            cache_lru.pop_front();
            cache_[idx] = cache_[to_remove];
            cache_[to_remove] = 0;
        } else {
            cache_[idx] = new double[N_];
            --cache_free_;
        }
        for (int i = 0; i != N_; ++i) {
            if (i == idx && dcache_) cache_[i][i] = dcache_[i];
            else if(i != idx && cache_[i]) cache_[idx][i] = cache_[i][idx];
            else cache_[idx][i] = do_kernel(idx,i);
        }
    } else {
        cache_lru.erase(cache_iter[idx]);
    }
    cache_lru.push_back(idx);
    cache_iter[idx]=prior(cache_lru.end());
    return cache_[idx];
}

double* KernelCache::get_diag() {
    if (!dcache_) {
        dcache_ = new double[N_];
        for (int i = 0; i != N_; ++i) {
            if (cache_[i]) dcache_[i] = cache_[i][i];
            else dcache_[i] = do_kernel(i,i);
        }
    }
    return dcache_;
}


/***
 * Returns the value of Kernel(X_i1, X_i2).
 * Uses the cache if possible, but does not update it.
 */
double KernelCache::kernel_apply(int i1, int i2) const {
    if (cache_[i1]) {
        assert(do_kernel(i1,i2) == cache_[i1][i2]);
        return cache_[i1][i2];
    }
    if (cache_[i2]) {
        assert(do_kernel(i1,i2) == cache_[i2][i1]);
        return cache_[i2][i1];
    }
    if (i1 == i2 && dcache_) {
        assert(do_kernel(i1,i2) == dcache_[i1]);
        return dcache_[i1];
    }
    return do_kernel(i1,i2);
}

class PyKernel : public KernelComputation {
        PyKernel(const PyKernel&);
    public:
        PyKernel(PyObject* X, PyObject* kernel, int N);
        ~PyKernel();
    protected:
        virtual double do_kernel(int i1, int i2) const;
    private:
        PyObject* const X_;
        PyObject* const pykernel_;
        const int N_;
};

PyKernel::PyKernel(PyObject* X, PyObject* kernel, int N):
    X_(X),
    pykernel_(kernel),
    N_(N)
    {
        Py_INCREF(X);
        Py_INCREF(kernel);
    }

PyKernel::~PyKernel() {
    Py_DECREF(X_);
    Py_DECREF(pykernel_);
}

double PyKernel::do_kernel(int i1, int i2) const {
    assert(i1 < N_);
    assert(i2 < N_);
    PyObject* obj1 = PySequence_GetItem(X_,i1);
    PyObject* obj2 = PySequence_GetItem(X_,i2);

    if (!obj1 || !obj2) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        throw SMO_Exception("svm.eval_SMO: Unable to access element in X array");
    }
    PyObject* arglist = Py_BuildValue("(OO)",obj1,obj2);
    PyObject* result = PyEval_CallObject(pykernel_,arglist);
    Py_XDECREF(obj1);
    Py_XDECREF(obj2);
    Py_DECREF(arglist);
    if (!result) { 
        check_for_interrupts();
        throw SMO_Exception("svm.eval_SMO: Unable to call kernel");
    }
    double val = PyFloat_AsDouble(result);
    Py_DECREF(result);
    return val;
}

class RBFKernel : public KernelComputation { 
    public:
        RBFKernel(PyArrayObject* X, double gamma);
        ~RBFKernel();
    protected:
        virtual double do_kernel(int i1, int i2) const;
    private:
        PyArrayObject* X_;
        double ngamma_;
        const int N_;
        const int N1_;
};

RBFKernel::RBFKernel(PyArrayObject* X, double gamma):
    X_(reinterpret_cast<PyArrayObject*>(X)),
    ngamma_(-1./gamma),
    N_(PyArray_DIM(X,0)),
    N1_(PyArray_DIM(X,1))
    {
        if (!PyArray_Check(X)) {
            throw SMO_Exception("RBF Kernel used, but not with numpy array.");
        }
        if (!PyArray_ISCARRAY(X)) {
            throw SMO_Exception("RBF Kernel used but not with CARRAY.");
        }
        Py_INCREF(X);
    }

RBFKernel::~RBFKernel() {
    Py_DECREF(X_);
}

double RBFKernel::do_kernel(int i1, int i2) const {
    assert(i1 < N_);
    assert(i2 < N_);
    const double* data1 = static_cast<const double*>(PyArray_GETPTR1(X_,i1));
    const double* data2 = static_cast<const double*>(PyArray_GETPTR1(X_,i2));
    double sumdiff = 0.;
    for (int i = 0; i != N1_; ++i) {
        double diff = data1[i]-data2[i];
        sumdiff += diff * diff;
    }
    sumdiff *= ngamma_;
    double res = std::exp(sumdiff);
    return res;
}

class DotKernel : public KernelComputation { 
    public:
        explicit DotKernel(PyArrayObject* X);
        ~DotKernel();
    protected:
        virtual double do_kernel(int i1, int i2) const;
    private:
        PyArrayObject* X_;
        const int N_;
        const int N1_;
};

DotKernel::DotKernel(PyArrayObject* X):
    X_(reinterpret_cast<PyArrayObject*>(X)),
    N_(PyArray_DIM(X,0)),
    N1_(PyArray_DIM(X,1))
    {
        if (!PyArray_Check(X)) {
            throw SMO_Exception("Dot Kernel used, but not with numpy array.");
        }
        if (!PyArray_ISCARRAY(X)) {
            throw SMO_Exception("Dot Kernel used but not with CARRAY.");
        }
        Py_INCREF(X);
    }

DotKernel::~DotKernel() {
    Py_DECREF(X_);
}

double DotKernel::do_kernel(int i1, int i2) const {
    assert(i1 < N_);
    assert(i2 < N_);
    const double* data1 = static_cast<const double*>(PyArray_GETPTR1(X_,i1));
    const double* data2 = static_cast<const double*>(PyArray_GETPTR1(X_,i2));
    double dotsum = 0.;
    for (int i = 0; i != N1_; ++i) {
        dotsum += data1[i] * data2[i];
    }
    return dotsum;
}

class PrecomputedKernel : public KernelComputation { 
    public:
        PrecomputedKernel(PyArrayObject* X);
        ~PrecomputedKernel();
    protected:
        virtual double do_kernel(int i1, int i2) const;
    private:
        PyArrayObject* X_;
};

PrecomputedKernel::PrecomputedKernel(PyArrayObject* X):
    X_(X)
    {
        if (!PyArray_Check(X)) {
            throw SMO_Exception("PrecomputedKernel used, but not with numpy array.");
        }
        if (!PyArray_ISCARRAY(X)) {
            throw SMO_Exception("Precomputed used but not with CARRAY.");
        }
        Py_INCREF(X);
    }

PrecomputedKernel::~PrecomputedKernel() {
    Py_DECREF(X_);
}

double PrecomputedKernel::do_kernel(int i1, int i2) const {
    const double* data = static_cast<const double*>(PyArray_GETPTR2(X_,i1,i2));
    return *data;
}

std::auto_ptr<KernelComputation> get_kernel(PyObject* X, PyObject* kernel) {
    typedef std::auto_ptr<KernelComputation> res_type;
    if (PyCallable_Check(kernel)) return res_type(new PyKernel(X, kernel, PySequence_Length(X)));
    if (!PyTuple_Check(kernel) || PyTuple_Size(kernel) != 2) throw SMO_Exception("Cannot parse kernel.");
    PyObject* type = PyTuple_GET_ITEM(kernel,0);
    PyObject* arg = PyTuple_GET_ITEM(kernel,1);
    if (!PyInt_Check(type) || !PyFloat_Check(arg)) throw SMO_Exception("Cannot parse kernel (wrong types)");
    long type_nr = PyInt_AsLong(type);
    double arg_value = PyFloat_AsDouble(arg);
    switch (type_nr) {
        case 0:
            return res_type(new RBFKernel(reinterpret_cast<PyArrayObject*>(X), arg_value));
        case 1:
            return res_type(new PrecomputedKernel(reinterpret_cast<PyArrayObject*>(X)));
        case 2:
            return res_type(new DotKernel(reinterpret_cast<PyArrayObject*>(X)));
        default:
            throw SMO_Exception("Unknown kernel type!");
    }
}


class LIBSVM_KernelCache : public KernelCache {
    public:
        LIBSVM_KernelCache(const int* Y, std::auto_ptr<KernelComputation> kernel, int N, int cache_nr_doubles)
            :KernelCache(kernel,N,cache_nr_doubles),
             Y_(Y)
             {  }
    private:
        double do_kernel(int i, int j) const {
            double res = KernelCache::do_kernel(i,j);
            return res * Y_[i] * Y_[j];
        }
        const int* Y_;
};

class SMO {
    public:
        SMO(PyObject* X, int* Y, double* Alphas, double& b, double C, int N, PyObject* kernel, double eps, double tol, int cache_size):
            Alphas(Alphas),
            Y(Y),
            b(b),
            C(C),
            N(N),
            cache_(get_kernel(X,kernel),N,cache_size),
            eps(eps),
            tol(tol) { }
        void optimise();
        double apply(int) const;
        bool take_step(int,int);
        bool examine_example(int);
        double get_error(int) const;

    private:
        double* Alphas;
        int* Y;
        double& b;
        double C;
        int N;
        mutable KernelCache cache_;
        const double eps;
        const double tol;
};

double SMO::get_error(int j) const {
    return apply(j) - Y[j];
}

double SMO::apply(int j) const {
    double sum = -b;
    double* Kernel_Line = cache_.get_kline(j);
    for (int i = 0; i != N; ++i) {
        if (Alphas[i] != C) {
            sum += Y[i] * Alphas[i] * Kernel_Line[i];
        }
    }
    //std::cout << "SMO::apply( " << j << " ): " << sum << '\n';
    return sum;
}
bool SMO::take_step(int i1, int i2) {
    //std::cout << "take_step( " << i1 << ", " << i2 << " );\n";
    if (i1 == i2) return false;
    const double alpha1 = Alphas[i1];
    const double alpha2 = Alphas[i2];
    const int y1 = Y[i1];
    const int y2 = Y[i2];
    double L, H;
    if (y1 != y2) {
        L = max(0.,alpha2-alpha1);
        H = min(C,C+alpha2-alpha1);
    } else {
        L = max(0.,alpha1+alpha2-C);
        H = min(C,alpha1+alpha2);
    }
    if (L == H) return false;
    const int s = y1*y2;
    const double E1 = get_error(i1);
    const double E2 = get_error(i2);
    const double k11 = cache_.kernel_apply(i1,i1);
    const double k12 = cache_.kernel_apply(i1,i2);
    const double k22 = cache_.kernel_apply(i2,i2);
    const double eta = 2*k12-k11-k22;
    double a1,a2;
    if (eta < 0) {
        a2 = alpha2-y2*(E1-E2)/eta;
        a2 = median(L,a2,H);
    } else {
        double gamma = alpha1+s*alpha2; // Eq. (12.22)
        double v1=E1+y1+b-y1*alpha1*k11-y2*alpha2*k12; // Eq. (12.21) // Note that f(x1) = E1 + y1
        double v2=E2+y2+b-y1*alpha1*k12-y2*alpha2*k22; // Eq. (12.21)
        double L_obj = gamma-s*L+L-.5*k11*(gamma-s*L)*(gamma-s*L)-.5*k22*L*L-s*k12*(gamma-s*L)*L-y1*(gamma-s*L)*v1-y2*L*v2; // # + W_const # Eq. (12.23)
        double H_obj = gamma-s*H+H-.5*k11*(gamma-s*H)*(gamma-s*L)-.5*k22*H*H-s*k12*(gamma-s*H)*H-y1*(gamma-s*H)*v1-y2*H*v2; // # + W_const # Eq. (12.23)
        if (L_obj > H_obj + eps) {
            a2 = L;
        } else if (L_obj < H_obj - eps) {
            a2 = H;
        } else {
            a2 = alpha2;
        }
    }
    if (a2 < tol) a2 = 0;
    else if (a2 > C-tol) a2 = C;
    if (std::abs(a2-alpha2) < eps*(a2+alpha2+eps)) return false;

    a1 = alpha1+s*(alpha2-a2);
    if (a1 < tol) a1 = 0;
    if (a1 > C-tol) a1 = C;

    // update everything
    Alphas[i1]=a1;
    Alphas[i2]=a2;
    double b1 = E1 + Y[i1]*(a1-alpha1)*k11+Y[i2]*(a2-alpha2)*k12+b; // Eq. (12.9)
    double b2 = E2 + Y[i1]*(a1-alpha1)*k12+Y[i2]*(a2-alpha2)*k22+b; // Eq. (12.10)
    const double new_b = (b1+b2)/2.;
    /*
    #for i in xrange(N):
    #    if Alphas[i] in (0,C):
    #        continue
    #    elif i == i1 or i == i2:
    #        E[i] = 0
    #    else:
    #        E[i] += y1*(a1-alpha1)*kernel_apply(i1,i)+y2*(a2-alpha2)*kernel_apply(i2,i) + (b-new_b) # Eq. (12.11)
    #E[i1]=f_at(i1)-y1
    #E[i2]=f_at(i2)-y2
    */
    b = new_b;
    return true;
}

bool SMO::examine_example(int i2) {
    //std::cout << "examine_example( " << i2 << " ) " << std::endl;
    const int y2 = Y[i2];
    const double alpha2 = Alphas[i2];
    const double E2 = get_error(i2);
    const double r2 = E2 * y2;
    //#print 'alpha2', alpha2, 'E2', E2, 'r2', r2
    if ( ( (r2 < -tol) && (alpha2 < C) ) ||
        ( (r2 > tol) && (alpha2 > 0) )){
        int best_i1 = -1;
        double bestE = -1;

        for (int i = 0; i != N; ++i) {
            if (Alphas[i] != 0 && Alphas[i] != C) {
                double dE = E2-get_error(i);
                if (dE < 0.) dE = -dE;
                if (dE > bestE) {
                    bestE=dE;
                    best_i1 = i;
                }
            }
        }
        if (best_i1 != -1 && take_step(best_i1,i2)) return true;
        for (int i1 = 0; i1 != N; ++i1) {
            if (Alphas[i1] && Alphas[i1] != C && take_step(i1,i2)) return true;
        }
        for (int i1 = 0; i1 != N; ++i1) {
            if (take_step(i1,i2)) return true;
        }
    }
    return false;
}

void SMO::optimise() {
    b = 0;
    for (int i = 0; i != N; ++i) Alphas[i] = 0;
    int changed = 0;
    bool examineAll = true;
    //int iter = 0;
    while (changed || examineAll) {
        //std::cout << "SMO::optimize loop: " << iter++ << "\n";
        check_for_interrupts();
        changed = 0;
        for (int i = 0; i != N; ++i) {
            if (examineAll || (Alphas[i] != 0 && Alphas[i] != C)) {
                changed += examine_example(i);
            }
        }
        if (examineAll) examineAll = false;
        else if (!changed) examineAll = true;
    }
}

void assert_type_contiguous(PyArrayObject* array,int type) { 
    if (!PyArray_Check(array) ||
        PyArray_TYPE(array) != type ||
        !PyArray_ISCONTIGUOUS(array)) {
        throw SMO_Exception("Arguments to eval_(SMO|LIBSVM) don't conform to expectation. Are you calling this directly? This is an internal function!");
    }
}

PyObject* eval_SMO(PyObject* self, PyObject* args) {
    try {
        PyObject* X;
        PyArrayObject* Y;
        PyArrayObject* Alphas0;
        PyArrayObject* params;
        PyObject* kernel;
        int cache_size;
        if (!PyArg_ParseTuple(args, "OOOOOi", &X, &Y, &Alphas0, &params,&kernel,&cache_size)) {
            const char* errmsg = "Arguments were not what was expected for eval_SMO.\n" 
                                "This is an internal function: Do not call directly unless you know what you're doing.\n";
            PyErr_SetString(PyExc_RuntimeError,errmsg);
            return 0;
        }
        assert_type_contiguous(Y,NPY_INT32);
        assert_type_contiguous(Alphas0,NPY_DOUBLE);
        assert_type_contiguous(params,NPY_DOUBLE);
        if (PyArray_DIM(params,0) < 4) throw SMO_Exception("eval_SMO: Too few parameters");

        int * Yv = static_cast<int*>(PyArray_DATA(Y));
        double* Alphas = static_cast<double*>(PyArray_DATA(Alphas0));
        unsigned N = PyArray_DIM(Y,0);
        double b = *static_cast<double*>(PyArray_GETPTR1(params,0));
        double C = *static_cast<double*>(PyArray_GETPTR1(params,1));
        double eps = *static_cast<double*>(PyArray_GETPTR1(params,2));
        double tol = *static_cast<double*>(PyArray_GETPTR1(params,3));
        SMO optimiser(X,Yv,Alphas,b,C,N,kernel,eps,tol,cache_size);
        optimiser.optimise();
        *static_cast<double*>(PyArray_GETPTR1(params,0)) = b; // Write back b
        Py_RETURN_NONE;
    } catch (const Python_Exception&) {
        // if Python_Exception was thrown, then PyErr is already set.
        return 0;
    } catch (const SMO_Exception& exc) {
        PyErr_SetString(PyExc_RuntimeError,exc.msg);
        return 0;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,"Some sort of exception in eval_SMO.");
        return 0;
    }
}


// The code for LIBSVM_Solver is taken from LIBSVM and adapted to work well in milk
// Copyright (c) 2000-2008 Chih-Chung Chang and Chih-Jen Lin
// Changes were made to make it more similar to our formulation.

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//

#define info printf
void info_flush() { }

class LIBSVM_Solver {
    public:
        LIBSVM_Solver(PyObject* X, int* Y, double* Alphas, double* p, double& b, double C, int N, PyObject* kernel, double eps, double tol, int cache_size, bool shrinking):
            Alphas(Alphas),
            Y(Y),
            b(b),
            C(C),
            N(N),
            cache_(Y,get_kernel(X,kernel),N,cache_size),
            eps(eps),
            tol(tol),
            alpha_status(N),
            active_size(N),
            active_set(N),
            p(p),
            shrinking(shrinking),
            tau(eps)
            {  }
        virtual ~LIBSVM_Solver() { }

        struct SolutionInfo {
            double obj;
            double rho;
            double upper_bound_p;
            double upper_bound_n;
            double r;	// for LIBSVM_Solver_NU
        };

        void optimise();
    protected:
        double* Alphas;
        int* Y;
        double& b;
        double C;
        const int N;
        LIBSVM_KernelCache cache_;
        const double eps;
        const double tol;
        vector<double> G;		// gradient of objective function
        enum alpha_status_e { lower_bound, upper_bound, free};
        vector<alpha_status_e> alpha_status;
        int active_size;
        vector<int> active_set;
        double *p;
        vector<double> G_bar;		// gradient, if we treat free variables as 0
        bool unshrinked;	// XXX
        bool shrinking;
        double tau;

        double get_C(int i) const {
            return C;
            //return (Y[i] > 0)? Cp : Cn;
        }
        void update_alpha_status(int i)
        {
            if(Alphas[i] >= get_C(i))
                alpha_status[i] = upper_bound;
            else if(Alphas[i] <= 0)
                alpha_status[i] = lower_bound;
            else alpha_status[i] = free;
        }
        bool is_upper_bound(int i) const { return alpha_status[i] == upper_bound; }
        bool is_lower_bound(int i) const { return alpha_status[i] == lower_bound; }
        bool is_free(int i) const { return alpha_status[i] == free; }
        void swap_index(int i, int j);
        void reconstruct_gradient();
        virtual bool select_working_set(int &i, int &j);
        virtual double calculate_rho();
        virtual void do_shrinking();
    private:
        bool be_shrunken(int i, double Gmax1, double Gmax2);
        void print_status() const;	
};

void LIBSVM_Solver::swap_index(int i, int j)
{
    // We *do not* swap in the cache or kernel
    // Therefore *all acesses* to the cache or kernel
    // must be of the form cache_.get_kernel(active_set[i])
    // instead of cache_.get_kernel(i)!
	std::swap(Y[i],Y[j]);
	std::swap(G[i],G[j]);
	std::swap(alpha_status[i],alpha_status[j]);
	std::swap(Alphas[i],Alphas[j]);
	std::swap(p[i],p[j]);
	std::swap(active_set[i],active_set[j]);
	std::swap(G_bar[i],G_bar[j]);
}

void LIBSVM_Solver::reconstruct_gradient()
{
	// reconstruct inactive elements of G from G_bar and free variables

	if(active_size == N) return;

	for(int i=active_size;i<N;++i)
		G[i] = G_bar[i] + p[i];
	
	for(int i=0;i<active_size;++i) {
		if(is_free(i))
		{
			const double *Q_i = cache_.get_kline(active_set[i],N);
            const double alpha_i = Alphas[i];
			for(int j=active_size;j<N;j++)
				G[j] += alpha_i * Q_i[active_set[j]];
		}
    }
}

void LIBSVM_Solver::optimise()  {
    // INITIALISE:
	unshrinked = false;

    for(int i=0;i<N;i++) {
        update_alpha_status(i);
        active_set[i] = i;
    }
    active_size = N;

	// initialize gradient
	G.resize(N);
	G_bar.resize(N);
	for(int i=0;i<N;++i) {
		G[i] = p[i];
		G_bar[i] = 0;
	}
	for(int i=0;i<N;i++) {
		if(!is_lower_bound(i))
		{
			const double *Q_i= cache_.get_kline(active_set[i]);
			const double alpha_i = Alphas[i];
			for(int j=0;j<N;j++) G[j] += alpha_i*Q_i[j];
			if(is_upper_bound(i)) {
				for(int j=0;j<N;j++) G_bar[j] += get_C(i) * Q_i[j];
            }
		}
	}

    //MAIN LOOP
	int counter = min(N,1000)+1;
    const int max_iters = 10*1000;

    for (int iter = 0; iter != max_iters; ++iter) {
        if (!(iter % 16)) check_for_interrupts();
		// show progress and do shrinking
		if(--counter == 0) {
			counter = min(N,1000);
			if(shrinking) do_shrinking();
			//info("."); info_flush();
		}

		int i,j;
		if(select_working_set(i,j)) {
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = N;
			//info("*"); info_flush();
			if(select_working_set(i,j)) {
                break;
            }
			else counter = 1;	// do shrinking next iteration
		}
		
		assert((i >= 0) && (i < active_size));
		assert((j >= 0) && (j < active_size));

		// update alpha[i] and alpha[j], handle bounds carefully
		
		const double *Q_i = cache_.get_kline(active_set[i], active_size);
		const double *Q_j = cache_.get_kline(active_set[j], active_size);

		const double C_i = get_C(i);
		const double C_j = get_C(j);

		const double old_alpha_i = Alphas[i];
		const double old_alpha_j = Alphas[j];

		if(Y[i]!=Y[j]) {
			double quad_coef = Q_i[active_set[i]]+Q_j[active_set[j]]+2.0*Q_i[active_set[j]];
			if (quad_coef <= 0) quad_coef = tau;
			const double delta = (-G[i]-G[j])/quad_coef;
			const double diff = Alphas[i] - Alphas[j];
			Alphas[i] += delta;
			Alphas[j] += delta;
			
			if(diff > 0) {
				if(Alphas[j] < 0) {
					Alphas[j] = 0;
					Alphas[i] = diff;
				}
			} else {
				if(Alphas[i] < 0) {
					Alphas[i] = 0;
					Alphas[j] = -diff;
				}
			}

			if(diff > C_i - C_j) {
				if(Alphas[i] > C_i) {
					Alphas[i] = C_i;
					Alphas[j] = C_i - diff;
				}
			} else {
				if(Alphas[j] > C_j) {
					Alphas[j] = C_j;
					Alphas[i] = C_j + diff;
				}
			}
            //print_status();
		} else {
			double quad_coef = Q_i[active_set[i]]+Q_j[active_set[j]]-2*Q_i[active_set[j]];
			if (quad_coef <= 0) quad_coef = tau;
			const double delta = (G[i]-G[j])/quad_coef;
			double sum = Alphas[i] + Alphas[j];
			Alphas[i] -= delta;
			Alphas[j] += delta;

			if(sum > C_i) {
				if(Alphas[i] > C_i) {
					Alphas[i] = C_i;
					Alphas[j] = sum - C_i;
				}
			} else {
				if(Alphas[j] < 0) {
					Alphas[j] = 0;
					Alphas[i] = sum;
				}
			}

			if(sum > C_j) {
				if(Alphas[j] > C_j) {
					Alphas[j] = C_j;
					Alphas[i] = sum - C_j;
				}
			} else {
				if(Alphas[i] < 0) {
					Alphas[i] = 0;
					Alphas[j] = sum;
				}
			}
		}

		// update G

		const double delta_Alphas_i = Alphas[i] - old_alpha_i;
		const double delta_Alphas_j = Alphas[j] - old_alpha_j;
		
        // print_status();
		for(int k=0;k<active_size;++k) {
			G[k] += Q_i[active_set[k]]*delta_Alphas_i + Q_j[active_set[k]]*delta_Alphas_j;
		}

		// update Alphas_status and G_bar
		const bool ui = is_upper_bound(i);
		const bool uj = is_upper_bound(j);
		update_alpha_status(i);
		update_alpha_status(j);
		if(ui != is_upper_bound(i)) {
			Q_i = cache_.get_kline(active_set[i], N);
			if(ui) {
				for(int k=0;k<N;k++)
					G_bar[k] -= C_i * Q_i[active_set[k]];
            } else {
				for(int k=0;k<N;k++)
					G_bar[k] += C_i * Q_i[active_set[k]];
            }
		}

		if(uj != is_upper_bound(j)) {
			Q_j = cache_.get_kline(active_set[j], N);
			if(uj) {
				for(int k=0;k<N;k++)
					G_bar[k] -= C_j * Q_j[active_set[k]];
            } else {
				for(int k=0;k<N;k++)
					G_bar[k] += C_j * Q_j[active_set[k]];
            }
		}
	}





    // CLEANUP


	// calculate rho

	//si->rho = calculate_rho();
	b = calculate_rho();

	// calculate objective value
    double v = 0;
    for(int i=0;i<N;i++) {
        v += Alphas[i] * (G[i] + p[i]);
    }

    //si->obj = v/2;

	// put back the solution
    for (int i = 0; i != N; ++i) {
        while (active_set[i] !=  i) {
            int j = active_set[i];
            std::swap(Y[i],Y[j]); // It's not polite to clobber Y, so put it back
            std::swap(Alphas[i],Alphas[j]);
            std::swap(active_set[i],active_set[j]);
        }
    }


	//si->upper_bound_p = Cp;
	//si->upper_bound_n = Cn;

}

void LIBSVM_Solver::print_status() const {
    std::cout << "    active_set_size: " <<  active_size << "\n";
    for (int i = 0; i != N; ++i) {
        std::cout << "    active_set[i]: " << active_set[i] << "\n";
        std::cout << "    p[i]: " << p[i] << "\n";
        std::cout << "    G[i]: " << G[i] << "\n";
        std::cout << "    G_bar[i]: " << G_bar[i] << "\n";
        std::cout << "    alpha_status[i]: " << alpha_status[i] << "\n";
        std::cout << "    Y[i]: " << Y[i] << "\n";
        std::cout << "    Alphas[i]: " << Alphas[i] << "\n";
        std::cout << "\n";
    }
}
// return true if already optimal, return false otherwise
bool LIBSVM_Solver::select_working_set(int &out_i, int &out_j) {
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

    //print_status();
	
	double Gmax = -INF;
	double Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++) {
		if( (Y[t] == +1 && !is_upper_bound(t)) ||
            (Y[t] == -1 && !is_lower_bound(t)))	{
            if (-Y[t]*G[t] >= Gmax) {
                Gmax = -Y[t]*G[t];
                Gmax_idx = t;
            }
        }
    }

    if (Gmax_idx == -1) return true;
	const int i = Gmax_idx;
	const double* Q_i = cache_.get_kline(active_set[i], N);
    const double* QDiag = cache_.get_diag();

	for(int j=0;j<active_size;++j) {
        if ((Y[j] == +1 && !is_lower_bound(j)) ||
            (Y[j] == -1 && !is_upper_bound(j))) {
                const double YGj = Y[j]*G[j];
                const double grad_diff = Gmax+YGj;
                if (YGj >= Gmax2) Gmax2 = YGj;
                if (grad_diff > 0) {
                    const double quad_coef = Q_i[active_set[i]] + QDiag[active_set[j]] - 2*Y[j]*Y[i]*Q_i[active_set[j]];
                    const double obj_diff_factor = (quad_coef > 0) ? quad_coef : tau;
                    const double obj_diff = -(grad_diff*grad_diff)/obj_diff_factor;

                    if (obj_diff <= obj_diff_min) {
                        Gmin_idx = j;
                        obj_diff_min = obj_diff;
                    }
                }
        }
	}

    if (Gmin_idx == -1) return true;
	if(Gmax+Gmax2 < eps) return true;
	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return false;
}

bool LIBSVM_Solver::be_shrunken(int i, const double Gmax1, const double Gmax2)
{
	if(is_upper_bound(i)) {
		if(Y[i]==+1) return -G[i] > Gmax1;
        return -G[i] > Gmax2;
	} else if(is_lower_bound(i)) {
		if (Y[i]==+1) return G[i] > Gmax2;
        return G[i] > Gmax1;
	}
    return false;
}

void LIBSVM_Solver::do_shrinking()
{
	double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(int i=0;i<active_size;i++) {
		if(Y[i]==+1)	{
			if(!is_upper_bound(i)) Gmax1 = max(-G[i],Gmax1);
			if(!is_lower_bound(i)) Gmax2 = max( G[i],Gmax2);
		} else	{
			if(!is_upper_bound(i)) Gmax2 = max(-G[i],Gmax2);
			if(!is_lower_bound(i)) Gmax1 = max( G[i],Gmax1);
		}
	}

	// shrink
	for(int i=0;i<active_size;++i) {
		if (be_shrunken(i, Gmax1, Gmax2)) {
			--active_size;
			while (active_size > i) {
				if (!be_shrunken(active_size, Gmax1, Gmax2)) {
					swap_index(i,active_size);
					break;
				}
				--active_size;
			}
		}
    }

	// unshrink, check all variables again before final iterations

	if(unshrinked || Gmax1 + Gmax2 > eps*10) return;
	
	unshrinked = true;
	reconstruct_gradient();

	for(int i= N-1; i >= active_size; --i) {
		if (!be_shrunken(i, Gmax1, Gmax2)) {
			while (active_size < i) {
				if (be_shrunken(active_size, Gmax1, Gmax2)) {
					swap_index(i,active_size);
					break;
				}
				++active_size;
			}
			++active_size;
		}
    }
}

double LIBSVM_Solver::calculate_rho() {
	int nr_free = 0;
	double ub = INF, lb = -INF, sum_free = 0;
	for(int i=0;i<active_size;i++) {
		const double yG = Y[i]*G[i];
		if(is_upper_bound(i))
		{
			if(Y[i]==-1) ub = min(ub,yG);
			else lb = max(lb,yG);
		}
		else if(is_lower_bound(i))
		{
			if(Y[i]==+1) ub = min(ub,yG);
			else lb = max(lb,yG);
		}
		else
		{
			++nr_free;
			sum_free += yG;
		}
	}

	if(nr_free>0) return sum_free/nr_free;
	return (ub+lb)/2;
}


PyObject* eval_LIBSVM(PyObject* self, PyObject* args) {
    try {
        PyObject* X;
        PyArrayObject* Y; 
        PyArrayObject* Alphas0;
        PyArrayObject* p; 
        PyArrayObject* params; 
        PyObject* kernel;
        int cache_size;
        if (!PyArg_ParseTuple(args, "OOOOOOi", &X, &Y, &Alphas0, &p, &params,&kernel,&cache_size)) {
            const char* errmsg = "Arguments were not what was expected for eval_LIBSVM.\n" 
                                "This is an internal function: Do not call directly unless you know exactly what you're doing.\n";
            PyErr_SetString(PyExc_RuntimeError,errmsg);
            return 0;
        }
        assert_type_contiguous(Y,NPY_INT32);
        assert_type_contiguous(Alphas0,NPY_DOUBLE);
        assert_type_contiguous(p,NPY_DOUBLE);
        assert_type_contiguous(params,NPY_DOUBLE);
        if (PyArray_DIM(params,0) < 4) throw SMO_Exception("eval_LIBSVM: Too few parameters");
        int * Yv = static_cast<int*>(PyArray_DATA(Y));
        double* Alphas = static_cast<double*>(PyArray_DATA(Alphas0));
        double* pv = static_cast<double*>(PyArray_DATA(p));
        unsigned N = PyArray_DIM(Y,0);
        double& b = *static_cast<double*>(PyArray_GETPTR1(params,0));
        double C = *static_cast<double*>(PyArray_GETPTR1(params,1));
        double eps = *static_cast<double*>(PyArray_GETPTR1(params,2));
        double tol = *static_cast<double*>(PyArray_GETPTR1(params,3));
        LIBSVM_Solver optimiser(X,Yv,Alphas,pv,b,C,N,kernel,eps,tol,cache_size, true);
        optimiser.optimise();
        Py_RETURN_NONE;
    } catch (const Python_Exception&) {
        // if Python_Exception was thrown, then PyErr is already set.
        return 0;
    } catch (const SMO_Exception& exc) {
        PyErr_SetString(PyExc_RuntimeError,exc.msg);
        return 0;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError,"Some sort of exception in eval_LIBSVM.");
        return 0;
    }
}

PyMethodDef methods[] = {
  {"eval_SMO",eval_SMO, METH_VARARGS , "Do NOT call directly.\n" },
  {"eval_LIBSVM",eval_LIBSVM, METH_VARARGS , "Do NOT call directly.\n" },
  {NULL, NULL,0,NULL},
};

const char  * module_doc = 
    "Internal SVM Module.\n"
    "\n"
    "Do NOT use directly!\n";

} // namespace

extern "C"
void init_svm()
  {
    import_array();
    (void)Py_InitModule3("_svm", methods, module_doc);
  }

