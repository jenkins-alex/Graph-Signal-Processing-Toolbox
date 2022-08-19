from test.test_graph_processes import test_pure_AR
from models.autoregressive.autoregressive import AdaptiveGraphAR, CausalGraphProcess
from datasets.simulated_data.simulated_data import arbitrary_graph_pure_AR, \
    create_random_matrix_as_in_paper, create_random_hs_as_in_paper
from datasets.graph_processes.graph_processes import GraphPureAR
import matplotlib.pyplot as plt
import numpy as np

def extract_features(data, P):
    features = []
    for p in range(1, P+1):
        f = np.roll(data, shift=p, axis=0)
        features.append(f)

    features = np.array(features)
    return features

def test_adaptive():
    N = 12
    P = 3
    data, generator = arbitrary_graph_pure_AR(N, P, 1000)
    
    alpha = 1
    mus = np.exp(np.linspace(4, -3, num=P))
    gamma = 10
    zeta = 5
    X = extract_features(data, P)
    y = data
    # gar = GraphAR(X, y, N, P, alpha, mus, zeta, gamma=5, init_type='rand')
    # gar.fit(max_iter=100)
    agar = AdaptiveGraphAR(P, N, alpha, mus, gamma,stepsize_filter=1e-4, stepsize_weight_matrix=1e-4, stepsize_debiasing=1e-4, time_switch_algorithms=400)
    agar.fit(X, y)
    plt.imshow(generator.weight_matrix)
    plt.show()
    plt.imshow(agar.W)
    plt.show()

def test_cgp():
    N = 30
    P = 3
    process_length = 1000
    burn_in = 500
    set_seed = False
    mc_repititions = 20
    max_iter = 10

    # define weight matrix and filter coefficients for graph process
    weight_matrix = create_random_matrix_as_in_paper(N)
    filter_coefficients = create_random_hs_as_in_paper(P)
    
    # get estimations from data
    weight_matrices = [generate_data_and_estimate_adjacency(P, N, process_length, set_seed, burn_in, max_iter, weight_matrix, filter_coefficients) for i in range(mc_repititions)]
    
    # monte carlo estimate of weight matrix from 10 repitions
    weight_matrix_estimate = np.stack(weight_matrices).mean(axis=0)

    # plot results
    plt.imshow(weight_matrix)
    plt.show()
    plt.imshow(weight_matrix_estimate)
    plt.show()

def generate_data_and_estimate_adjacency(P, N, process_length, set_seed, burn_in, max_iter, weight_matrix, filter_coefficients):
        # generate data from graph stochastic process
        while True:
            pure_ar = GraphPureAR(weight_matrix, process_length, set_seed, P, filter_coefficients=filter_coefficients)
            data = pure_ar.simulate_data_from_initial
            data = data[burn_in:]  # use only data after initial burn in 
            if data.max() < 100:
                print('stable data found!')
                break

        # training
        alpha = 1
        mus = 3
        gamma = 10
        zeta = 5
        X = extract_features(data, P)
        y = data
        gar = CausalGraphProcess(X, y, N, P, alpha, mus, zeta, gamma=gamma, init_type='rand', skip_step_2=True)
        gar.fit(max_iter=max_iter)
        return gar.W

if __name__ == '__main__':
    test_cgp()
    # test_adaptive()
    