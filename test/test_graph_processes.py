from re import A
import numpy as np
from datasets.graph_processes.graph_processes import GraphPureAR
import matplotlib.pyplot as plt

def test_pure_AR():
    # artbitrarily random weight matrix
    adj_matrix = np.zeros(shape=(5, 5))
    matrix_coords = np.reshape(np.arange(adj_matrix.size), (5, 5))
    indices = np.random.choice(
        np.arange(adj_matrix.size),
        replace=False,
        size=int(adj_matrix.size * 0.2))
    adj_matrix = np.isin(matrix_coords, indices).astype(int)
    
    # normalise the adj_matrix to stop data blowing up
    D = np.diag(np.sum(adj_matrix, axis=0))  # degree matrix
    normaliser = np.linalg.pinv(D)**0.5
    weight_matrix = np.matmul(normaliser, np.matmul(adj_matrix, normaliser))
    
    process_length = 100
    set_seed = True
    auto_reg_terms = 3
    filter_coefficients = None
    pure_ar = GraphPureAR(weight_matrix, process_length, set_seed, auto_reg_terms, filter_coefficients=filter_coefficients)
    outs = pure_ar.simulate_data_from_initial
    
    # plot 
    plt.plot(outs)
    plt.show()
