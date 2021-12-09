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
        return self.W > 0

    def _create_graph(self):
        # create the weighted graph using networkX
        return nx.from_numpy_matrix(self.W)
    
    def _get_edges(self):
        return self.graph.edges()

    def _get_degree_matrix(self):
        degree_list = [val for (node, val) in self.graph.degree()]
        return np.identity(len(degree_list)) * degree_list
    
    def _get_laplacian(self, norm=False):
        if norm:
            return nx.linalg.laplacianmatrix.normalized_laplacian_matrix(self.graph).toarray()
        return self.D - self.A

    def eig_decompose(self):
        # compute eigenvalues and eigenvectors, w and v, respectively
        w, v = np.linalg.eig(self.L)

        # the first eigenvalue is a maximally smooth constant, can omit first of both 
        w = w[1:]
        v = v[:, 1:]
        return w, v


class Student:
    
    def __init__(self, exam_types):
        # assign random preferred exam type out of exam_types possible types
        self.preferred_type = np.random.randint(exam_types, size=1)[0]
    
    def sit_exams(self, exams, default_mean=65, pref_mean=75, default_std=10, pref_std=10):
        # compute the mean value of Gaussian distribution for exam grades for student
        mean_exams = np.zeros(shape=exams.shape)
        mean_exams[exams == self.preferred_type] = pref_mean
        mean_exams[exams != self.preferred_type] = default_mean
        
        # compute the std value of Gaussian distribution for student
        std_exams = np.zeros(shape=exams.shape)
        std_exams[exams == self.preferred_type] = pref_std
        std_exams[exams != self.preferred_type] = default_std
        
        # randomly sample grades from Gaussian distribution with mean and std
        exam_grades = np.random.normal(loc=mean_exams, scale=std_exams)
        
        # ensure exam results are between 0 and 100
        exam_grades[exam_grades > 100.0] = 100.0
        exam_grades[exam_grades < 0.0] = 0.0
        self.exam_grades = exam_grades
        
    def get_exam_grades(self):
        return self.exam_grades
    
    def get_type(self):
        return self.preferred_type


def create_exams(total_exams, types=3):
    # create array of exam types of length equal to the number of total exams
    return np.random.randint(types, size=total_exams)


if __name__ == '__main__':

    np.random.seed(21)
    total_exams = 40
    total_students = 70
    exam_types = 3
    exams = create_exams(total_exams, types=exam_types)
    
    # create a NxM matrix of exam results, where N is the number of students, and M is each exam.
    exam_results = []
    preferred_types = []
    for student in range(0, total_students):
        student = Student(exam_types)
        student.sit_exams(exams)
        exam_results.append(student.get_exam_grades())
        preferred_types.append(student.get_type())
        
    # create matrix
    X = np.vstack(exam_results)
    true_labels = np.array(preferred_types)

    sc = SpectralClustering(X, norm=True, kernel='rbf', gamma=None, edge_thresh=None)
    w, v = sc.eig_decompose()
    print(w)
    print(v)