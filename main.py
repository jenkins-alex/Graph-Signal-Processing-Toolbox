from test.test_graph_processes import test_pure_AR
from models.autoregressive.autoregressive import AdaptiveGraphAR, CausalGraphProcess
from datasets.simulated_data.simulated_data import arbitrary_graph_pure_AR
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
    agar = AdaptiveGraphAR(P, N, alpha, mus, gamma, stepsize_filter=1e-4, stepsize_weight_matrix=1e-4, stepsize_debiasing=1e-4, time_switch_algorithms=400)
    agar.fit(X, y)
    plt.imshow(generator.weight_matrix)
    plt.show()
    plt.imshow(agar.W)
    plt.show()

def test_cgp():
    N = 12
    P = 3
    data, generator = arbitrary_graph_pure_AR(N, P, 1000)
    
    alpha = 1
    mus = 3
    gamma = 10
    zeta = 5
    X = extract_features(data, P)
    y = data
    gar = CausalGraphProcess(X, y, N, P, alpha, mus, zeta, gamma=5, init_type='rand')
    gar.fit(max_iter=10)
    plt.imshow(generator.weight_matrix)
    plt.show()
    plt.imshow(gar.W)
    plt.show()

if __name__ == '__main__':
    test_cgp()
    # test_adaptive()
    