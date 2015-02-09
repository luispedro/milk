import numpy.random
import milk.unsupervised.pca
import numpy as np

def test_pca():
    numpy.random.seed(123)
    X = numpy.random.rand(10,4)
    X[:,1] += numpy.random.rand(10)**2*X[:,0] 
    X[:,1] += numpy.random.rand(10)**2*X[:,0] 
    X[:,2] += numpy.random.rand(10)**2*X[:,0] 
    Y,V = milk.unsupervised.pca(X)
    Xn = milk.unsupervised.normalise.zscore(X)
    assert X.shape == Y.shape
    assert ((np.dot(V[:4].T,Y[:,:4].T).T-Xn)**2).sum()/(Xn**2).sum() < .3

def test_mds():
    from milk.unsupervised import pdist
    np.random.seed(232)
    for _ in range(12):
        features = np.random.random_sample((12,4))
        X = milk.unsupervised.mds(features,4)
        D = pdist(features)
        D2 = pdist(X)
        assert np.mean( (D - D2) ** 2) < 10e-4


def test_mds_dists():
    from milk.unsupervised import pdist
    np.random.seed(232)
    for _ in range(12):
        features = np.random.random_sample((12,4))
        D = pdist(features)
        X = milk.unsupervised.mds(features,4)
        X2 = milk.unsupervised.mds_dists(D, 4)
        assert np.mean( (X - X2) ** 2) < 10e-4



def test_mds_list():
    from milk.unsupervised.pca import mds
    data = np.random.random((128,16))
    V  = mds(data,2)
    V2 = mds(list(data),2)
    assert np.all(V == V2)

def test_mds_regression_eig_order():
    from milk.unsupervised.pca import mds_dists
    # This was part of a much larger computation, but this isolated the bug:
    dists = np.array([[
                  0.        ,  377241.01101501,  390390.47006156,
             340764.02535826,  421258.30020762,  470960.15365819,
             331864.64507197,  213029.60122458,  306976.87583849],
           [ 377241.01101501,       0.        ,  159390.25449606,
             140506.60640227,  140922.67044651,  221684.10621381,
             130161.14561428,  224134.4629224 ,  225617.6525412 ],
           [ 390390.47006156,  159390.25449606,       0.        ,
             188417.11617804,  192114.58972062,  238026.3963446 ,
             159070.76483779,  242792.81436928,  228843.70200362],
           [ 340764.02535826,  140506.60640227,  188417.11617804,
                  0.        ,  247098.49216397,  265783.27794352,
             161672.29500768,  170503.64299615,  171360.11464776],
           [ 421258.30020762,  140922.67044651,  192114.58972062,
             247098.49216397,       0.        ,  246385.36543382,
             153380.00248566,  276707.33890808,  276009.04198403],
           [ 470960.15365819,  221684.10621381,  238026.3963446 ,
             265783.27794352,  246385.36543382,       0.        ,
             252609.80940353,  327987.54137854,  308492.70255307],
           [ 331864.64507197,  130161.14561428,  159070.76483779,
             161672.29500768,  153380.00248566,  252609.80940353,
                  0.        ,  179275.66833105,  192598.94271197],
           [ 213029.60122458,  224134.4629224 ,  242792.81436928,
             170503.64299615,  276707.33890808,  327987.54137854,
             179275.66833105,       0.        ,  117004.41340669],
           [ 306976.87583849,  225617.6525412 ,  228843.70200362,
             171360.11464776,  276009.04198403,  308492.70255307,
             192598.94271197,  117004.41340669,       0.        ]])
    V = milk.unsupervised.mds_dists(dists, 2)
    assert V[:,1].ptp() > 1.
