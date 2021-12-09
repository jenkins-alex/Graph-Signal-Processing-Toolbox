import numpy as np
import networkx as nx
import itertools as it


class SpectralClustering:

    def __init__(self, X, norm=True, kernel='rbf', gamma=None, edge_thresh=None):
        """initalise class

        Args:
            X (np.array): 2d numpy array (samples x features)
            norm (bool): to use normalised graph Laplacian in spectral clustering
            kernel (str): kernel to use from possible in ['rbf']
            gamma (float): constant value to use in rbf kernel
            edge_thresh (float): threshold value, define edge between 
                two nodes for euclid distances less than this value 
        """
        self.X = X
        self.kernel = kernel
        self.gamma = gamma
        self.edge_thresh = edge_thresh
        self.W = self._construct_weight_matrix()
        self.A = self._get_adjacency()
        self.graph = self._create_graph()
        self.edges = self._get_edges()
        self.D = self._get_degree_matrix()
        self.L = self._get_laplacian(norm=norm)

    def _symmetrise(self, arr):
        """ Return a symmetrised version of numpy array, arr.

        Values 0 are replaced by the array value at the symmetric
        position (with respect to the diagonal), i.e. if a_ij = 0,
        then the returned array a' is such that a'_ij = a_ji.

        Diagonal values are left untouched.

        a -- square NumPy array, such that a_ij = 0 or a_ji = 0, 
        for i != j.

        Args:
            a (np.array): 2d numpy array to make symmetric

        Returns:
            np.array: symmetric 2d numpy array
        """
        return arr + arr.T - np.diag(arr.diagonal())

    def _find_pairs(self):
        """ construct list of all pairs of samples """
        return list(it.combinations(range(0, self.X.shape[0]), 2))

    def _construct_weight_matrix(self):
        """ construct weight matrix from data points using kernel

        Returns:
            np.array: weight matrix for graph
        """

        self.pairs = self._find_pairs()

        total_samples = self.X.shape[0]
        W = np.zeros(shape=(total_samples,total_samples))  # compute empty weight matrix
        ds = []
        for pair in self.pairs:
            id_A = pair[0]
            id_B = pair[1]
            sample_A = self.X[id_A, :]
            sample_B = self.X[id_B, :]
            d = np.linalg.norm(sample_A-sample_B)
            ds.append(d)
            if self.kernel == 'rbf':
                W[id_A][id_B] = self._rbf_kernel(d)
        W = self._symmetrise(W)
        return W

    def _rbf_kernel(self, d):
        """ compute radial basis function for distance

        Args:
            d (float): euclidean distance between two points

        Returns:
            float: radial basis weighted distance
        """
        if self.gamma is None:
            self.gamma = 1 / self.X.shape[1]

        if self.edge_thresh is None:
            return np.exp(-1 * self.gamma * d**2)

        if d < self.edge_thresh:
            return np.exp(-1 * self.gamma * d**2)
        return 0.0
        
    def _get_adjacency(self):
        """ find the adjacency matrix for graph

        Returns:
            np.array: graph adjacency matrix
        """
        return self.W > 0

    def _create_graph(self):
        """ create networkx graph object

        Returns:
            nx.Graph: networkx graph object from weight matrix
        """
        # create the weighted graph using networkX
        return nx.from_numpy_matrix(self.W)
    
    def _get_edges(self):
        """ edges of graph

        Returns:
            nx.EdgeView: graph edges
        """
        return self.graph.edges()

    def _get_degree_matrix(self):
        """ compute node degree matrix

        Returns:
            np.array: degree matrix
        """
        degree_list = [val for (node, val) in self.graph.degree()]
        return np.identity(len(degree_list)) * degree_list
    
    def _get_laplacian(self, norm=False):
        """ compute graph Laplacian

        Args:
            norm (bool, optional): whether to use standard or 
                normalised graph Laplacian matrix. Defaults to False.

        Returns:
            np.array: graph Laplacian matrix
        """
        if norm:
            return nx.linalg.laplacianmatrix.normalized_laplacian_matrix(self.graph).toarray()
        return self.D - self.A

    def eig_decompose(self):
        """ compute eigenvalues and eigenvectors of graph Laplacian

        Returns:
            tuple of np.array: (eigenvectors, eigenvalues)
        """
        # compute eigenvalues and eigenvectors, w and v, respectively
        w, v = np.linalg.eig(self.L)

        # the first eigenvalue is a maximally smooth constant, can omit first of both 
        w = w[1:]
        v = v[:, 1:]
        return w, v
