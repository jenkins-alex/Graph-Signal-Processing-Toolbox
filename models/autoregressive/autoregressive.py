import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from signals.filters import PolynomialGraphFilter
from signals.graph_shift_operators import GraphShiftOperator
from scipy.optimize import minimize

class GraphAR:

    def __init__(self, X, y, N, P, alpha, mu, gamma=None, init_type='rand', concat=None):
        """
        Batch vertex-time graph auto-regressive model, as seen in, https://arxiv.org/abs/2003.05729.
        Here the graph shift operator and the graph filter coefficients are learnt from data.
        The optimal coefficients are found for the whole dataset. This model is not adaptive.

        Args:
            X (np.array): Auto-regressive features (P x M x N)
            y (np.array): Vector of labels to be predicted (M x N)
            N (int): Number of outputs to be predicted.
            P (int): Number of auto-regressive terms.
            alpha (int): Sample weighting for each time-step [0, 1].
            mu (np.array): Vector of l1 regularisation strengths (P).
            gamma (float, optional): Regularisation strength for commutivity term.
                Defaults to None (no commutivity term).
            init_type (str, optional): Weight initalisiation 'rand' or 'zeros'. Defaults to 'rand'.
            concat (str, optional): Method to use for many-to-one prediction merging, 'sum' or 'mean'. 
                Defaults to None (many-to-many prediction).
        """
        self.X = X  # features (P previous time steps) PxMxN where M is batch size
        self.y = y  # labels MxN
        self.N = N  # number of nodes in graph
        self.P = P  # order of graph filter
        self.alpha = alpha  # int weighting of time-steps [0, 1]
        self.mu = mu  # vector of l1 regularisation strengths of size P
        self.gamma = gamma  # float value for regularisation on commutivity term
        self.smoothness_reg = smoothness_reg  # float value for smoothness regularisation of graph signal
        
        # specify if commutativity term should exist in loss function
        if self.gamma is None:
            self.add_commutivity_term = False
        else:
            self.add_commutivity_term = True

        # initialise the learnable filter weights
        if init_type == 'rand':
            self.beta = np.random.rand(P, N, N)  # initialise filter model parameters
        else:
            self.beta = np.zeros(shape=(P, N, N))  # initialise parameters to zero
        
        # concatonation method for predictions
        self.concat = concat
        
    def fit(self, method='BFGS', max_iter=1):
        """ learn parameters of the graph auto-regression model

        Args:
            method (str, optional): Optimizer to use. Defaults to 'BFGS'.
            max_iter (int, optional): Max iterations to use. Defaults to 1.
        """
        self._learn_filter(method, max_iter)
        #self._learn_gso(X, y)
        #self._learn_filter_coefs(X, y)
        pass
    
    def predict(self, X):
        """ predict the output for input features X

        Args:
            X (np.array): input features

        Returns:
            np.array: vector of outputs for given inputs 
        """
        # calculate graph filter terms
        prediction = np.matmul(self.beta, np.swapaxes(X, 1, 2)).sum(axis=0).T
        return prediction
    
    def get_gso(self):
        """ approximate the graph shift operator (GSO) as the first component of the 
        learnt weights (when considering a polynomial graph filter, the first component
        of the learnt graph filter is proportional to the GSO)

        Returns:
            np.array: approximation of the GSO
        """
        # as the first component of beta weights is a linear function of the gso,
        # the first component can be a good approximation of the gso
        return np.copy(self.beta[0, :, :])
    
    def visualise_filter(self):
        """plot the learnt graph filters as heatmaps
        """
        for i in range(0, self.P):
            sns.heatmap(self.beta[i])
            plt.show()

    def _learn_filter(self, method, max_iter):
        """learn filters using gradient descent

        Args:
            method (str): optimisation method to use. 
            max_iter (int): Max iterations to use.
        """
        res = minimize(self._filter_loss_function,
                       self.beta,
                       method=method,
                       options={'maxiter': max_iter})
    
    def _filter_loss_function(self, beta):
        """
        The loss function for the graph auto-regressive model. Corresponds to equation 11
        in https://arxiv.org/abs/2003.05729, except an additional smoothness term can be potentially added,
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7032244.

        Args:
            beta (np.array): Learnable weights for each graph filter

        Returns:
            float: value of loss at given iteration 
        """
        
        # update weights
        self.beta = beta.reshape(self.P, self.N, self.N)
        
        # calculate MSE for each time
        MSE_term = np.linalg.norm(self.y-self.predict(self.X), ord=2, axis=1)**2

        # sum the MSE over all time-steps
        powers = np.array([self.y.shape[0] - i for i in range(0, self.y.shape[0])])
        alphas = np.array([self.alpha] * self.y.shape[0])
        powers = np.power(alphas, powers)
        weighted_mses = np.multiply(powers, MSE_term)
        loss = .5 * np.sum(weighted_mses)
        
        # add in the sparsity term for the parameters
        l1_norms = np.linalg.norm(self.beta, ord=1, axis=(1,2))
        l1_loss = np.sum(np.multiply(self.mu, l1_norms))
        loss += l1_loss
        
        # add in the commutivity term
        if self.add_commutivity_term:
            # calculate each combination of multiplied filter terms  
            arr = np.dot(self.beta, self.beta).reshape(self.P, self.P, self.N, self.N)
            comm_terms = []
            for i in range(0, self.P):
                for j in range(0, self.P):
                    if i == j:
                        continue
                    comm_terms.append(np.linalg.norm(arr[i,j] - arr[j,i], 'fro')**2)
            loss += self.gamma * np.sum(comm_terms)
        print(loss)
        return loss


class AdaptiveGraphAR:
    
    def __init__(self, P, N, alpha, mus, gamma, stepsize):
        """
        Adaptive vertex-time graph auto-regressive model, as seen in, https://arxiv.org/abs/2003.05729.
        Here the graph shift operator and the graph filter coefficients are learnt from data.

        Args:
            P (int): Number of auto-regressive terms.
            N (int): Number of vertices of graph / sensor channels.
            alpha (float): Sample weighting for each time-step [0, 1].
            mus (np.array): Vector of l1 regularisation strengths (P).
            gamma (float): Regularisation strength for commutivity term.
            stepsize (float): Step-size for gradient based projection updates.
        """
        # initialise attributes
        self.P = P  # number of filter terms
        self.N = N  # number of vertices of graph
        self.alpha = alpha  # regularisation
        self.mus = mus  # regularisation for sparsity
        self.gamma = gamma  # regularisation for commutativity
        self.stepsize = stepsize  # stepsize for adaptive updates
        
        # initialise learnable parameters
        self.beta_pos = np.zeros(shape=(N, N*P))
        self.beta_neg = np.zeros(shape=(N, N*P))
        self.p_matrix = np.zeros(shape=(N, N*P))
        self.r_matrix = np.zeros(shape=(N*P, N*P))
        
        # matrix of stepsizes
        self.A = self.stepsize * np.eye(P)
        
        # put mus in matrix 
        self.M = np.array([self.mus[p] * np.ones(shape=(self.N, self.N)) for p in range(0, self.P)])
        self.M = self.M.T.reshape(self.M.shape[-1], -1)  # make shape NxNP
        
    def predict(self, X):
        """
        Predict the output / labels for input features X

        Args:
            X (np.array): input features over time

        Returns:
            np.array: array of predicted output / labels over time
        """

        # get adaptive predictions for each time step of X
        predictions = []
        for t in range(0, X.shape[1]):   
            # get features for t
            xPt = X[:, t, :].flatten()
            xPt = xPt.reshape(xPt.shape[0], 1) 
            
            # get filter term for t
            fPt = self.filters[t, :, :]
            
            # get predictions for t
            yt = np.matmul(fPt, xPt)
            predictions.append(yt)
        
        predictions = np.stack(predictions)  # T x N x 1
        predictions = predictions.reshape(
            predictions.shape[0], predictions.shape[1])  # T x N
        return predictions
    
    def fit(self, X, y):
        """
        Train the model using the loss function for the graph auto-regressive model given by
        equation 11 in https://arxiv.org/abs/2003.05729, by a series of gradient projection 
        updates given in the paper's section 4.

        Args:
            X (np.array): input auto-regressive features for the model
            y (np.array): output / training labels over time.

        Returns:
            np.array: learnt graph filters at each time-step
        """
        filters = []
        for t in tqdm(range(0, X.shape[1])):
            
            xPt = X[:, t, :].flatten()
            xPt = xPt.reshape(xPt.shape[0], 1)            
            self.r_matrix = self.alpha * self.r_matrix + np.dot(xPt, xPt.T)
            yt = y[t, :]
            yt = yt.reshape(yt.shape[0], 1)
            self.p_matrix = self.alpha * self.p_matrix + np.dot(yt, xPt.T)
            
            # learnable filter terms NxNP
            self.beta = np.subtract(self.beta_pos, self.beta_neg)

            # commutivity terms NxNP
            self.Q = np.array([self.commutivity_loss_term(p) for p in range(0, self.P)])  # PxNxN
            self.Q = self.Q.T.reshape(self.Q.shape[-1], -1)  # make shape NxNP
            
            # TODO: add smoothness terms NxNP
            
            # compute G
            self.G = np.matmul(self.beta, self.r_matrix) - (self.p_matrix - self.gamma * self.Q)  # NxNP
            
            # update learnable terms as a gradient projection
            self.beta_pos = self.beta_pos - np.matmul((self.M+self.G), np.kron(self.A, np.eye(self.N)))
            self.beta_neg = self.beta_neg - np.matmul((self.M-self.G), np.kron(self.A, np.eye(self.N)))

            # keep only positive parts of the matrices
            self.beta_pos[self.beta_pos < 0] = 0
            self.beta_neg[self.beta_neg < 0] = 0

            # update parameter matrix
            self.beta = np.subtract(self.beta_pos, self.beta_neg)
            filters.append(self.beta)
            
        filters = np.stack(filters)  # T x N x NP
        self.filters = filters
        return filters

    def commutivity_loss_term(self, p):
        """
        The commutation term for the loss function.

        Args:
            p (int): current time-step

        Returns:
            np.array: commutivity loss matrix
        """
        # function to compute Qp,t+1

        beta_p = self.beta.reshape(self.N, self.N, self.P)[:, :, p] # NxN
        q_terms = []
        for k in range(2, self.P):
            beta_k = self.beta.reshape(self.N, self.N, self.P)[:, :, k]
            first_term = np.matmul(self.commute_loss(beta_p, beta_k), beta_k.T)
            second_term = np.matmul(beta_k.T, self.commute_loss(beta_p, beta_k))
            q_terms.append(first_term - second_term)
        return np.stack(q_terms).sum(axis=0)
        
    def commute_loss(self, m1, m2):
        """
        the commutivity penalty for two matrices, m1 and m2.

        Args:
            m1 (np.array): first matrix
            m2 (np.array): second matrix

        Returns:
            np.array: commutivity penalties
        """
        return np.matmul(m1, m2) - np.matmul(m2, m1) 