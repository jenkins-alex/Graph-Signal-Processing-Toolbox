import numpy as np
from datasets.graph_processes.graph_processes import GraphPureAR

def generate_data_from_pure_AR_model(N, auto_reg_terms, process_length):
    # artbitrarily random graph
    adj_matrix = np.zeros(shape=(N, N))
    matrix_coords = np.reshape(np.arange(adj_matrix.size), (N, N))
    indices = np.random.choice(
        np.arange(adj_matrix.size),
        replace=False,
        size=int(adj_matrix.size * 0.2))
    adj_matrix = np.isin(matrix_coords, indices).astype(int)
    print(adj_matrix)
    
    # normalise the adj_matrix to stop data blowing up
    D = np.diag(np.sum(adj_matrix, axis=0))  # degree matrix
    normaliser = np.linalg.pinv(D)**0.5
    weight_matrix = np.matmul(normaliser, np.matmul(adj_matrix, normaliser))
    set_seed = True
    filter_coefficients = None
    pure_ar = GraphPureAR(weight_matrix, process_length, set_seed, auto_reg_terms, filter_coefficients=filter_coefficients)
    outs = pure_ar.simulate_data_from_initial
    return outs