import numpy as np
from graph_shift_operators import GraphShiftOperator

class PolynomialGraphFilter:
    def __init__(self, S, K, h=None):
        """polynomial graph filter

        Args:
            S (np.ndarray): shift matrix for graph shift operator (NxN)
            K (int): order of polynomial graph filter
            h (np.array, optional): set of weights for each filter term. Defaults to None.
        """
        self.S = S
        self.K = K
        self.h = h
        
    def filt(self, x):
        """apply filter to signal across vertices of graph

        Args:
            x (np.array): signal to filter

        Returns:
            np.array: weighted / unweighted graph filter terms
        """
        gso = GraphShiftOperator(self.S)

        terms = []
        terms.append(x)  # add the 0th term - data remains unchanged
        for k in range(1, self.K+1):
            term = np.matmul(gso.power(k), x)
            terms.append(term)
        term_stack = np.hstack(terms)
        
        if self.h is None:
            # return unweighted filter terms
            return term_stack

        # return weighted filter terms
        hs = np.repeat(self.h, int(len(term_stack)/len(self.h)))
        return np.multiply(hs, term_stack)