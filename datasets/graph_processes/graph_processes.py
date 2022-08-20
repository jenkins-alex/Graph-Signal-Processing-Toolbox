# -*- coding: utf-8 -*-

"""simulated data from different graph processes using arbitrarily defined graphs
"""
import numpy as np
from abc import ABC, abstractmethod
from signals.filters import PolynomialGraphFilter

class SimulatedGraphProcess(ABC):
    """Abstract base class for simulated graph process

    Args:
        ABC (object): class of abstract base class
    """
    def __init__(self, weight_matrix, process_length, set_seed):
        """

        Args:
            weight_matrix (np.array): NxN matrix where N is number of nodes in graph
            process_length (int): number of time steps to simulate for process
            set_seed (bool): set numpy random seed.
        """
        self.weight_matrix = weight_matrix
        self.process_length = process_length
        self.set_seed = set_seed
        if self.set_seed:
            np.random.seed(42)

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

    def __init__(self, weight_matrix, process_length, set_seed, auto_reg_terms, filter_coefficients=None):
        super().__init__(weight_matrix, process_length, set_seed)
        """initialisation of graph AR base class

        Args:
            weight_matrix (np.array): weight matrix
            process_length (int): number of time steps to simulate
            set_seed (bool): set numpy random seed.
            auto_reg_terms (int): number of AR terms to use, referred to as P
            filter_coefficients (np.array, optional): P*(P+1) vector. Defaults to None.
        """
        self.auto_reg_terms = auto_reg_terms  # number of AR terms to use in process
        self.initialise_filter_coefficients(filter_coefficients)

    @abstractmethod
    def graph_process(self, data):
        """abstract method for the definition of the graph process equation

        Args:
            data (np.array): the data to process using graph process
        """
        pass

    @property
    def get_filter_coefs(self):
        """
        Returns:
            int: np.array the filter coefficients used in simulation
        """
        return self.filter_coefficients

    @property
    def simulate_data_from_initial(self):
        """simulated data using defined auto-regressive graph process
        """
        # calculate number of nodes in graph
        N = self.weight_matrix.shape[0]

        # initialise input data
        initial_data = np.random.normal(np.zeros(N), np.ones(N)) # array size N

        # initialise containers
        outputs = np.zeros(shape=(N, self.process_length))  # outputs of graph AR process
        outputs[:, 0] = initial_data
        auto_reg_data = np.zeros(shape=(self.auto_reg_terms, N))  # input data to AR graph process
        auto_reg_data[0] = initial_data

        # simulation
        # TODO: vectorise simulated data curation
        for i in range(1, self.process_length):
            # get data at next time-step 
            output = self.graph_process(auto_reg_data) + np.random.normal(np.zeros(N), np.ones(N)) # size N
            outputs[:, i] = output  # save in container

            # update data for AR
            auto_reg_data = np.roll(auto_reg_data, 1, axis=0)  # roll data back 1 step in time
            auto_reg_data[0] = output # add output as new time step

        # return converted output as T x N array
        return outputs.T

    def initialise_filter_coefficients(self, filter_coefficients):
        """randomly and sparsely initialise filter coefficients if not provided
        """
        # set the filter coefficients as random and sparse if not provided
        if filter_coefficients is None:
            M = int(self.auto_reg_terms * (self.auto_reg_terms+3) / 2)  # number of filter coefficients
            new_filter_coefficients = np.zeros(M)
            indices = np.random.choice(
                np.arange(new_filter_coefficients.size),
                replace=False,
                size=int(new_filter_coefficients.size * 0.2))
            new_filter_coefficients[indices] = 1  # set 20% of filter coefficients to 1, rest are zero
            self.filter_coefficients = new_filter_coefficients
        else:
            self.filter_coefficients = filter_coefficients


class GraphPureAR(GraphAR):
    """simulate data from a purely auto-regressive causal graph process

    Args:
        GraphAR (object): abstract base class for graph AR process
    """
    def __init__(self, weight_matrix, process_length, set_seed, auto_reg_terms, filter_coefficients, set_first_two_filter_coefficients=True):
        super().__init__(weight_matrix, process_length, set_seed, auto_reg_terms, filter_coefficients)
        self.set_first_two_filter_coefficients = set_first_two_filter_coefficients
        if self.set_first_two_filter_coefficients:
            # enforces first graph filter = graph shift operator
            # enables identifiability of graph shift operator from data
            self.filter_coefficients[0] = 0
            self.filter_coefficients[1] = 1

    def graph_process(self, X):
        """predict the next time-step using learnt GSO and filter coefficients

        Args:
            X (np.array): input features # P x N
        Returns:
            np.array: vector of outputs for given inputs 
        """
        # calculate the graph filtered data
        filtered_terms = []
        for i in range(1, self.auto_reg_terms+1):
            pgf = PolynomialGraphFilter(self.weight_matrix, i)
            filtered_terms.append(pgf.filt(X[i-1]).T)
        term_stack = np.concatenate(filtered_terms, axis=1)

        # weight terms using coefficients and calculate sum
        weighted_terms = np.reshape(self.filter_coefficients, (1, -1)) * term_stack
        predictions = np.sum(weighted_terms, axis=1)
        return predictions  # size N


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
    