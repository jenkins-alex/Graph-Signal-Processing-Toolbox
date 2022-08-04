"""simulated data from different graph processes using arbitrarily defined graphs
"""
import numpy as np
from abc import ABC, abstractmethod


class SimulatedGraphProcess(ABC):
    """Abstract base class for simulated graph process

    Args:
        ABC (object): class of abstract base class
    """
    def __init__(self, weight_matrix, process_length, order_graph_filter):
        """

        Args:
            weight_matrix (np.array): NxN matrix where N is number of nodes in graph
            process_length (int): number of time steps to simulate for process
            order_graph_filter (int): number of terms to use in polynomial graph filter
        """
        self.weight_matrix = weight_matrix
        self.process_length = process_length
        self.order_graph_filter = order_graph_filter

    @property
    def get_weight_matrix(self):
        """return weight matrix

        Returns:
            np.array: NxN weight matrix where N is the number of nodes
        """
        return self.weight_matrix

    @property
    def get_process_length(self):
        """return number of time steps of simulated data

        Returns:
            int: number of time steps
        """
        return self.process_length

    @abstractmethod
    def graph_process(self, data):
        """abstract method for the definition of the graph process equation

        Args:
            data (np.array): the data to process using graph process
        """
        pass

    @abstractmethod
    def simulate_data_from_initial(self):
        """abstract method for simulated data using defined graph process
        """
        pass


class GraphAR(SimulatedGraphProcess):

    def __init__(self, weight_matrix, process_length, order_graph_filter, auto_reg_terms):
        super().__init__(weight_matrix, process_length, order_graph_filter)
        self.auto_reg_terms = auto_reg_terms

    @abstractmethod
    def graph_process(self, data):
        """abstract method for the definition of the graph process equation

        Args:
            data (np.array): the data to process using graph process
        """
        pass

    @property
    def simulate_data_from_initial(self):
        """simulated data using defined auto-regressive graph process
        """
        # calculate number of nodes in graph
        N = self.weight_matrix.shape[0]

        # initialise input data
        initial_data = np.random.normal(np.zeros(N), np.eye(N))

        # initialise containers
        outputs = np.zeros(shape=(self.process_length, N, N))  # outputs of graph AR process
        outputs[0] = initial_data
        auto_reg_data = np.zeros(shape=(self.auto_reg_terms, N, N))  # input data to AR graph process
        auto_reg_data[0] = initial_data

        # simulation
        for i in range(1, self.process_length):
            # get data at next time-step 
            output = self.graph_process(auto_reg_data)  # NxN
            outputs[i] = output  # save in container

            # update data for AR
            auto_reg_data = np.roll(auto_reg_data, 1, axis=0)  # roll data back 1 step in time
            auto_reg_data[0] = output  # add output as new time step
        return outputs

class GraphPureAR(GraphAR):
    """simulate data from a purely auto-regressive graph process

    Args:
        SimulatedGraphProcess (object): abstract base class for graph process
    """
    def __init__(self, weight_matrix, process_length, order_graph_filter, auto_reg_terms):
        super().__init__(weight_matrix, process_length, order_graph_filter, auto_reg_terms)

    def graph_process(self, data):
        """definition of graph process to be used for simulation

        Args:
            data (np.array): PxNxN where P is the number of auto reg terms, N is number of nodes
        """
        pass


class GraphARMA(GraphAR):
    """simulate data from an ARMA process on graph

    Args:
        SimulatedGraphProcess (object): abstract base class for graph process
    """
    pass

class GraphARIMA(GraphAR):
    """simulate data from an ARIMA process on graph

    Args:
        SimulatedGraphProcess (object): abstract base class for graph process
    """
    pass
    