from milk.supervised.svm import svm_learn, svm_apply, svm_size
import numpy
eps=1e-3
def approximate(a,b):
    a=numpy.asanyarray(a)
    b=numpy.asanyarray(b)
    return numpy.abs(a-b).max() < eps

def assert_kkt(SVM):
    X,Y,Alphas,b,C,kernel=SVM
    N=len(Alphas)
    for i in xrange(N):
        if Alphas[i] == 0.:
            assert Y[i]*svm_apply(SVM,X[i])+eps >= 1
        elif Alphas[i] == C:
            assert Y[i]*svm_apply(SVM,X[i])-eps <= 1
        else:
            assert abs(Y[i]*svm_apply(SVM,X[i])-1) <= eps

def assert_all_correctly_classified(SVM,X,Y):
    N=svm_size(SVM)
    for i in xrange(N):
        assert svm_apply(SVM,X[i]) * Y[i] > 0

def assert_more_than_50(SVM,X,Y):
    N=svm_size(SVM)
    correct = 0
    for i in xrange(N):
        correct += (svm_apply(SVM,X[i]) * Y[i] > 0)
    assert correct > N/2

def test_simplest():
    X=numpy.array([
        [1],
        [2],
        ])
    Y=numpy.array([1,-1])
    C=4.
    kernel=numpy.dot
    Alphas,b=svm_learn(X,Y,kernel,C)
    SVM=(X,Y,Alphas,b,C,kernel)
    assert approximate(Alphas,[2.,2.])
    assert approximate(b,-3)
    assert_kkt(SVM)
    assert_all_correctly_classified(SVM,X,Y)

def test_more_complex():
    X=numpy.array([
        [1,0],
        [2,1],
        [2,0],
        [4,2],
        [0,1],
        [0,2],
        [1,2],
        [2,8]])
    Y=numpy.array([1,1,1,1,-1,-1,-1,-1])
    C=4.
    kernel=numpy.dot
    Alphas,b=svm_learn(X,Y,kernel,C)
    SVM=(X,Y,Alphas,b,C,kernel)

    sv=numpy.array([1,1,0,0,1,0,1,0])
    nsv=~sv
    computed_sv = (Alphas > 0) & (Alphas < C)
    computed_nsv = ~computed_sv
    assert_kkt(SVM)
    assert numpy.all((sv-computed_sv) >= 0) # computed_sv in sv
    assert numpy.all((computed_nsv-nsv) >= 0) # nsv in computed_nsv
    assert_all_correctly_classified(SVM,X,Y)

def rbf(xi,xj):
    return numpy.exp(-((xi-xj)**2).sum())
def test_rbf():
    X=numpy.array([
        [0,0,0],
        [1,1,1],
        ])
    Y=numpy.array([ 1, -1 ])
    C=10
    Alphas,b=svm_learn(X,Y,rbf,C)
    SVM=(X,Y,Alphas,b,C,rbf)
    assert_all_correctly_classified(SVM,X,Y)

def test_random():
    R=numpy.random.RandomState(123)
    X=R.rand(10,3)
    X[:5]+=.3
    C=2
    Y=numpy.ones(10)
    Y[5:] *= -1
    Alphas,b=svm_learn(X,Y,rbf,C)
    SVM=(X,Y,Alphas,b,C,rbf)
    assert_more_than_50(SVM,X,Y)

if __name__ == '__main__':
    test_simplest()
    test_more_complex()
    test_rbf()
    test_random()

