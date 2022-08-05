from test.test_graph_processes import test_pure_AR
from models.autoregressive.autoregressive import GraphAR, AdaptiveGraphAR
from datasets.simulated_data.simulated_data import arbitrary_graph_pure_AR
import numpy as np

def extract_features(data, P):
    features = []
    for p in range(1, P+1):
        f = np.roll(data, shift=p, axis=0)
        features.append(f)

    features = np.array(features)
    return features

if __name__ == '__main__':
    N = 12
    P = 3
    data, generator = arbitrary_graph_pure_AR(N, P, 250)
    print(data.shape)
    
    alpha = 1
    mus = np.exp(np.linspace(4, -3, num=P))
    gamma = 10
    zeta = 5
    X = extract_features(data, P)
    y = data
    # gar = GraphAR(X, y, N, P, alpha, mus, zeta, gamma=5, init_type='rand')
    # gar.fit(max_iter=100)
    agar = AdaptiveGraphAR(P, N, alpha, mus, gamma, stepsize_filter=1e-4, stepsize_weight_matrix=1e-4, stepsize_debiasing=1e-4)
    agar.fit(X, y)
    # print(np.round(generator.weight_matrix, 2))
    # print(np.round(gar.W, 2))
    # print('...')
    # print(generator.filter_coefficients)
    # print(gar.hs)
    # print('Done.')
    