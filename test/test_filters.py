import numpy as np
from signals.filters import PolynomialGraphFilter

def test_polynomial_graph_filter():
    """test to check if polynomial graph filtered data gives the right values
       test case for circularly connected path graph 
    """
    # circular graph shift operator
    A = np.diag(np.ones(4), 1)
    A[-1, 0] = 1
    
    # generate random graph signals
    x = np.random.rand(5, 10)
    
    # test graph shift operator works as expected
    assert np.all(np.matmul(A, x) == np.roll(x, -1, axis=0))
    assert np.all(np.matmul(A, np.matmul(A, x)) == np.roll(x, -2, axis=0))
    
    # test the polynomial filter class gives correct output
    pgf = PolynomialGraphFilter(A, 2)
    terms = pgf.filt(x)
    assert np.all(terms[0] == x)
    assert np.all(terms[1] == np.matmul(A, x))
    assert np.all(terms[2] == np.matmul(A, np.matmul(A, x)))
    
    print('[PASSED] Polynomial graph filter class.')
    