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
    labels=numpy.array([1,1,1,1,-1,-1,-1,-1])
    C=4.
    kernel=numpy.dot
    Alphas,b=svm_learn(X,labels,kernel,C)
    SVM=(X,labels,Alphas,b,C,kernel)

    sv=numpy.array([1,1,0,0,1,0,1,0])
    nsv=~sv
    computed_sv = (Alphas > 0) & (Alphas < C)
    computed_nsv = ~computed_sv
    assert_kkt(SVM)
    assert numpy.all((sv-computed_sv) >= 0) # computed_sv in sv
    assert numpy.all((computed_nsv-nsv) >= 0) # nsv in computed_nsv
    assert_all_correctly_classified(SVM,X,labels)

if __name__ == '__main__':
    test_simplest()
    test_more_complex()
