from test.test_graph_processes import test_pure_AR
from models.autoregressive.autoregressive import GraphAR
from datasets.simulated_data.simulated_data import arbitrary_graph_pure_AR

if __name__ == '__main__':
    data = arbitrary_graph_pure_AR(5, 3, 250)
    print(data.shape)

