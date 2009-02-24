import milk.measures.nfoldcrossvalidation
import numpy as np
def test_foldgenerator():
    labels = np.array([1]*20+[2]*30+[3]*20)
    for nf in [None,2,3,5,10,15,20]:
        assert np.array([test.copy() for _,test in milk.measures.nfoldcrossvalidation.foldgenerator(labels,nf)]).sum(0).max() == 1
        assert np.array([test.copy() for _,test in milk.measures.nfoldcrossvalidation.foldgenerator(labels,nf)]).sum() == len(labels)
        assert np.array([(test&train).sum() for train,test in milk.measures.nfoldcrossvalidation.foldgenerator(labels,nf)]).sum() == 0

