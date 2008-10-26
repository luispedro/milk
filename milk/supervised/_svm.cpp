#include <assert.h>
#include <cmath>
#include <iostream>
#include <list>
#include <vector>
extern "C" {
    #include <Python.h>
    #include <numpy/ndarrayobject.h>
}


namespace { 
/// This is a boost function
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
class KernelCache {
    public:
        KernelCache(PyObject* X, PyObject* kernel, int N, int cache_nr_doubles);
        ~KernelCache();
   
        double* get_kline(int idx);
        double kernel_apply(int i1, int i2);
    private:
        double do_kernel(int i1, int i2);

        PyObject* const X_;
        PyObject* const pykernel_;
        const int N_;
        double ** cache;
        int cache_free_;
        std::list<int> cache_lru;
        std::vector<std::list<int>::iterator> cache_iter;
};

KernelCache::KernelCache(PyObject* X, PyObject* kernel, int N, int cache_nr_floats):
    X_(X),
    pykernel_(kernel),
    N_(N) {
    cache = new double*[N_];
    for (int i = 0; i != N_; ++i) cache[i] = 0;
    cache_free_ = (cache_nr_floats/N_);
    cache_iter.resize(N_,cache_lru.end());
}

KernelCache::~KernelCache() {
    for (int i = 0; i != N_; ++i) delete [] cache[i];
    delete [] cache;
}

double* KernelCache::get_kline(int idx) {
    if (!cache[idx]) {
        if (!cache_free_) {
            int to_remove = cache_lru.front();
            cache_lru.pop_front();
            cache[idx] = cache[to_remove];
            cache[to_remove] = 0;
        } else {
            cache[idx] = new double[N_];
            --cache_free_;
        }
        for (int i = 0; i != N_; ++i) {
            if (i != idx && cache[i]) cache[idx][i] = cache[i][idx];
            else cache[idx][i] = do_kernel(idx,i);
        }
    } else {
        cache_lru.erase(cache_iter[idx]);
    }
    cache_lru.push_back(idx);
    cache_iter[idx]=prior(cache_lru.end());
    return cache[idx];
}

/***
 * Returns the value of Kernel(X_i1, X_i2).
 * Uses the cache if possible, but does not update it.
 */
double KernelCache::kernel_apply(int i1, int i2) {
    if (cache[i1]) {
        assert(do_kernel(i1,i2) == cache[i1][i2]);
        return cache[i1][i2];
    }
    if (cache[i2]) {
        assert(do_kernel(i1,i2) == cache[i2][i1]);
        return cache[i2][i1];
    }
    return do_kernel(i1,i2);
}

double KernelCache::do_kernel(int i1, int i2) {
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
        if (PyErr_Occurred()) {
            throw Python_Exception();
        }
        throw SMO_Exception("svm.eval_SMO: Unable to call kernel");
    }
    double val = PyFloat_AsDouble(result);
    Py_DECREF(result);
    return val;
}

class SMO {
    public:
        SMO(PyObject* X, int* Y, double* Alphas, double& b, double C, int N, PyObject* kernel, double eps, double tol, int cache_size):
            Alphas(Alphas),
            Y(Y),
            b(b),
            C(C),
            N(N),
            cache_(X,kernel,N,cache_size),
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
    if (abs(a2-alpha2) < eps*(a2+alpha2+eps)) return false;

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
    if ( (r2 < -tol) and (alpha2 < C) or (r2 > tol) and (alpha2 > 0)){
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
    while (changed || examineAll) {
        if (PyErr_Occurred()) throw Python_Exception();
        changed = 0;
        for (int i = 0; i != N; ++i) {
            if (examineAll || Alphas[i] != 0 && Alphas[i] != C) {
                changed += examine_example(i);
            }
        }
        if (examineAll) examineAll = false;
        else if (!changed) examineAll = true;
    }
}

void assert_type_contiguous(PyArrayObject* array,int type) { 
    if (!PyArray_TYPE(array) == type ||
        !PyArray_ISCONTIGUOUS(array)) {
        throw SMO_Exception("Arguments to eval_SMO don't conform.");
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
        assert_type_contiguous(Y,NPY_INT);
        assert_type_contiguous(Alphas0,NPY_DOUBLE);
        assert_type_contiguous(params,NPY_DOUBLE);
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
class Solver {
public:
	Solver() {};
	virtual ~Solver() {};

	struct SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
	};

	void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking);
protected:
	int active_size;
	schar *y;
	double *G;		// gradient of objective function
	enum { LOWER_BOUND, UPPER_BOUND, FREE };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double *alpha;
	const QMatrix *Q;
	const Qfloat *QD;
	double eps;
	double Cp,Cn;
	double *p;
	int *active_set;
	double *G_bar;		// gradient, if we treat free variables as 0
	int l;
	bool unshrinked;	// XXX

	double get_C(int i)
	{
		return (y[i] > 0)? Cp : Cn;
	}
	void update_alpha_status(int i)
	{
		if(alpha[i] >= get_C(i))
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
	bool is_free(int i) { return alpha_status[i] == FREE; }
	void swap_index(int i, int j);
	void reconstruct_gradient();
	virtual int select_working_set(int &i, int &j);
	virtual double calculate_rho();
	virtual void do_shrinking();
private:
	bool be_shrunken(int i, double Gmax1, double Gmax2);	
};

void Solver::swap_index(int i, int j)
{
	Q->swap_index(i,j);
	swap(y[i],y[j]);
	swap(G[i],G[j]);
	swap(alpha_status[i],alpha_status[j]);
	swap(alpha[i],alpha[j]);
	swap(p[i],p[j]);
	swap(active_set[i],active_set[j]);
	swap(G_bar[i],G_bar[j]);
}

void Solver::reconstruct_gradient()
{
	// reconstruct inactive elements of G from G_bar and free variables

	if(active_size == l) return;

	int i;
	for(i=active_size;i<l;i++)
		G[i] = G_bar[i] + p[i];
	
	for(i=0;i<active_size;i++)
		if(is_free(i))
		{
			const Qfloat *Q_i = Q->get_Q(i,l);
			double alpha_i = alpha[i];
			for(int j=active_size;j<l;j++)
				G[j] += alpha_i * Q_i[j];
		}
}

void Solver::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
{
	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_,l);
	clone(y, y_,l);
	clone(alpha,alpha_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	unshrinked = false;

	// initialize alpha_status
	{
		alpha_status = new char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for(int i=0;i<l;i++)
			active_set[i] = i;
		active_size = l;
	}

	// initialize gradient
	{
		G = new double[l];
		G_bar = new double[l];
		int i;
		for(i=0;i<l;i++)
		{
			G[i] = p[i];
			G_bar[i] = 0;
		}
		for(i=0;i<l;i++)
			if(!is_lower_bound(i))
			{
				const Qfloat *Q_i = Q.get_Q(i,l);
				double alpha_i = alpha[i];
				int j;
				for(j=0;j<l;j++)
					G[j] += alpha_i*Q_i[j];
				if(is_upper_bound(i))
					for(j=0;j<l;j++)
						G_bar[j] += get_C(i) * Q_i[j];
			}
	}

	// optimization step

	int iter = 0;
	int counter = min(l,1000)+1;

	while(1)
	{
		// show progress and do shrinking

		if(--counter == 0)
		{
			counter = min(l,1000);
			if(shrinking) do_shrinking();
			info("."); info_flush();
		}

		int i,j;
		if(select_working_set(i,j)!=0)
		{
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = l;
			info("*"); info_flush();
			if(select_working_set(i,j)!=0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}
		
		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully
		
		const Qfloat *Q_i = Q.get_Q(i,active_size);
		const Qfloat *Q_j = Q.get_Q(j,active_size);

		double C_i = get_C(i);
		double C_j = get_C(j);

		double old_alpha_i = alpha[i];
		double old_alpha_j = alpha[j];

		if(y[i]!=y[j])
		{
			double quad_coef = Q_i[i]+Q_j[j]+2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (-G[i]-G[j])/quad_coef;
			double diff = alpha[i] - alpha[j];
			alpha[i] += delta;
			alpha[j] += delta;
			
			if(diff > 0)
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = diff;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = -diff;
				}
			}
			if(diff > C_i - C_j)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = C_i - diff;
				}
			}
			else
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = C_j + diff;
				}
			}
		}
		else
		{
			double quad_coef = Q_i[i]+Q_j[j]-2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (G[i]-G[j])/quad_coef;
			double sum = alpha[i] + alpha[j];
			alpha[i] -= delta;
			alpha[j] += delta;

			if(sum > C_i)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = sum - C_i;
				}
			}
			else
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = sum;
				}
			}
			if(sum > C_j)
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = sum - C_j;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = sum;
				}
			}
		}

		// update G

		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;
		
		for(int k=0;k<active_size;k++)
		{
			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
		}

		// update alpha_status and G_bar

		{
			bool ui = is_upper_bound(i);
			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			update_alpha_status(j);
			int k;
			if(ui != is_upper_bound(i))
			{
				Q_i = Q.get_Q(i,l);
				if(ui)
					for(k=0;k<l;k++)
						G_bar[k] -= C_i * Q_i[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_i * Q_i[k];
			}

			if(uj != is_upper_bound(j))
			{
				Q_j = Q.get_Q(j,l);
				if(uj)
					for(k=0;k<l;k++)
						G_bar[k] -= C_j * Q_j[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_j * Q_j[k];
			}
		}
	}

	// calculate rho

	si->rho = calculate_rho();

	// calculate objective value
	{
		double v = 0;
		int i;
		for(i=0;i<l;i++)
			v += alpha[i] * (G[i] + p[i]);

		si->obj = v/2;
	}

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[active_set[i]] = alpha[i];
	}

	// juggle everything back
	/*{
		for(int i=0;i<l;i++)
			while(active_set[i] != i)
				swap_index(i,active_set[i]);
				// or Q.swap_index(i,active_set[i]);
	}*/

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	info("\noptimization finished, #iter = %d\n",iter);

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
	
	double Gmax = -INF;
	double Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)	
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax)
				{
					Gmax = -G[t];
					Gmax_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax = G[t];
					Gmax_idx = t;
				}
		}

	int i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i = Q->get_Q(i,active_size);

	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				double grad_diff=Gmax+G[j];
				if (G[j] >= Gmax2)
					Gmax2 = G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef=Q_i[i]+QD[j]-2*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff= Gmax-G[j];
				if (-G[j] >= Gmax2)
					Gmax2 = -G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef=Q_i[i]+QD[j]+2*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(Gmax+Gmax2 < eps)
		return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

bool Solver::be_shrunken(int i, double Gmax1, double Gmax2)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax2);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax1);
	}
	else
		return(false);
}

void Solver::do_shrinking()
{
	int i;
	double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(i=0;i<active_size;i++)
	{
		if(y[i]==+1)	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax1)
					Gmax1 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax2)
					Gmax2 = G[i];
			}
		}
		else	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax2)
					Gmax2 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax1)
					Gmax1 = G[i];
			}
		}
	}

	// shrink

	for(i=0;i<active_size;i++)
		if (be_shrunken(i, Gmax1, Gmax2))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunken(active_size, Gmax1, Gmax2))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}

	// unshrink, check all variables again before final iterations

	if(unshrinked || Gmax1 + Gmax2 > eps*10) return;
	
	unshrinked = true;
	reconstruct_gradient();

	for(i=l-1;i>=active_size;i--)
		if (!be_shrunken(i, Gmax1, Gmax2))
		{
			while (active_size < i)
			{
				if (be_shrunken(active_size, Gmax1, Gmax2))
				{
					swap_index(i,active_size);
					break;
				}
				active_size++;
			}
			active_size++;
		}
}

double Solver::calculate_rho()
{
	double r;
	int nr_free = 0;
	double ub = INF, lb = -INF, sum_free = 0;
	for(int i=0;i<active_size;i++)
	{
		double yG = y[i]*G[i];

		if(is_upper_bound(i))
		{
			if(y[i]==-1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else
		{
			++nr_free;
			sum_free += yG;
		}
	}

	if(nr_free>0)
		r = sum_free/nr_free;
	else
		r = (ub+lb)/2;

	return r;
}



PyMethodDef methods[] = {
  {"eval_SMO",eval_SMO, METH_VARARGS , "Do NOT call directly.\n" },
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

