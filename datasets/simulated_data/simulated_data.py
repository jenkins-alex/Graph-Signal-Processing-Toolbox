import numpy as np
from datasets.graph_processes.graph_processes import GraphPureAR

def create_random_normalised_weight_matrix(N):
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
    return weight_matrix

def create_random_matrix_as_in_paper(N):
    # as in Methods of Adaptive Signal Processing on Graphs Using Vertex-Time Autoregressive Models
    W = np.random.normal(0, 1, size=(N, N))
    
    # threshold weight matrix to between 0.3 and 0.7 of max weight
    max_weight = np.max(np.abs(W))
    W[np.abs(W)>0.7*max_weight] = 0
    W[np.abs(W)<0.3*max_weight] = 0
    
    # calculate the eigenvalues of W and normalise by 1.5x largest eigenvalue for stable process
    w, _ = np.linalg.eig(W)
    max_eig = np.max(np.abs(w.real))
    W /= 1.5 * max_eig
    return W
    
def create_random_hs_as_in_paper(P):
    """create random graph filter coefficients for simulation using method
    from the paper of Signal Processing on Graphs: Causal Modeling of
    Unstructured Data by Mei and Moura 2017:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7763882

    Args:
        P (int): number of auto-regressive terms

    Returns:
        np.array: filter coefficients to be used
    """
    M = int(P * (P+3) / 2)  # total number of filter coefficients
    
    # generate coefficients sparsely and from a mixture of uniform distributions
    filter_coefficients = 0.5 * np.random.uniform(-1, -0.45, M) + 0.5 * np.random.uniform(0.45, 1, M)
    filter_coefficients /= 1.5  # normalisation
    
    # multiply filter coefficients by 2^(i+j) where i refers to graph filter number, j to graph filter term
    # power_term_i_plus_j = []
    # for i in range(2, P+1):
    #     for j in range(0, i+1):
    #         power_term_i_plus_j.append(i+j)
    # power_term_i_plus_j = np.array(power_term_i_plus_j)
    # filter_coefficients[2:] /= 2**power_term_i_plus_j

    # set first two filter coefficients to avoid identifiability problems
    filter_coefficients[0] = 0
    filter_coefficients[1] = 1
    return filter_coefficients
    
def arbitrary_graph_pure_AR(N, auto_reg_terms, process_length):

    # weight_matrix = create_random_normalised_weight_matrix(N)
    weight_matrix = create_random_matrix_as_in_paper(N)
    set_seed = True
    filter_coefficients = create_random_hs_as_in_paper(auto_reg_terms)
    pure_ar = GraphPureAR(weight_matrix, process_length, set_seed, auto_reg_terms, filter_coefficients=filter_coefficients)
    outs = pure_ar.simulate_data_from_initial
    return outs, pure_ar