import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from signals.filters import PolynomialGraphFilter
from signals.graph_shift_operators import GraphShiftOperator
from datasets.graph_processes.graph_processes import GraphPureAR
from scipy.optimize import minimize

class GraphAR:
    """autoregressive model using graph filtering
    """
    def __init__(self, X, y, N, P, alpha, mu, zeta, gamma=None, init_type='rand'):
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
            zeta (float): Regularisation strength for sparsity of learnt filter coefficients.
            gamma (float, optional): Regularisation strength for commutivity term.
                Defaults to None (no commutivity term).
            init_type (str, optional): Weight initalisiation 'rand' or 'zeros'. Defaults to 'rand'.
        """
        self.X = X  # features (P previous time steps) PxMxN where M is batch size
        self.y = y  # labels MxN
        self.N = N  # number of nodes in graph
        self.P = P  # order of graph filter
        self.M = int(P*(P+3)/2) # number of filter coefficients
        self.alpha = alpha  # int weighting of time-steps [0, 1]
        self.mu = mu  # vector of l1 regularisation strengths of size P
        self.gamma = gamma  # float value for regularisation on commutivity term
        self.zeta = zeta  # regularisation strength for sparisty of learnt graph filter coefs

        # specify if commutativity term should exist in loss function
        if self.gamma is None:
            self.add_commutivity_term = False
        else:
            self.add_commutivity_term = True

        # initialise the learnable filter weights
        if init_type == 'rand':
            self.beta = np.random.rand(P, N, N)  # initialise filter model parameters
            self.W = np.random.rand(N, N)  # initialise the learnt GSO
            self.hs = np.random.rand(self.M)  # initialise the learnt filter coefficients
        else:
            self.beta = np.zeros(shape=(P, N, N))  # initialise parameters to zeros
            self.W = np.zeros(shape=(N, N))  # initialise the learnt GSO to zeros
            self.hs = np.zeros(self.M)  # initialise the learnt filter coefficients
        
    def fit(self, method='BFGS', max_iter=1):
        """ learn parameters of the graph auto-regression model

        Args:
            method (str, optional): Optimizer to use. Defaults to 'BFGS'.
            max_iter (int, optional): Max iterations to use. Defaults to 1.
        """
        print('(1/3) Learning graph filters from data...')
        self._learn_filter(method, max_iter)
        print('(2/3) Learning graph shift operator from graph filters...')
        self._learn_gso(method, max_iter)
        print('(3/3) Learning graph filter coefficients from data...')
        self._learn_filter_coefs(method, max_iter)
    
    def predict(self, X):
        """ predict the output for input features X

        Args:
            X (np.array): input features

        Returns:
            np.array: vector of outputs for given inputs 
        """
        # calculate graph filter terms
        # prediction = np.matmul(self.beta, np.swapaxes(X, 1, 2)).sum(axis=0).T
        prediction = np.zeros(shape=self.y.shape)
        for i in range(0, self.P):
            prediction += np.matmul(self.beta[i], X[i].T).T
        return prediction
    
    def get_approximate_gso(self):
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
                       self.beta.flatten(),
                       method=method,
                       options={'maxiter': max_iter})

    def _learn_gso(self, method, max_iter):
        """learn gso using gradient descent

        Args:
            method (str): optimisation method to use.
            max_iter (int): Max iterations to use.
        """
        res = minimize(self._gso_loss_function,
                       self.W.flatten(),
                       method=method,
                       options={'maxiter': max_iter})

    def _learn_filter_coefs(self, method, max_iter):
        """learn filter coefficients using gradient descent

        Args:
            method (str): optimisation method to use.
            max_iter (int): Max iterations to use.
        """
        res = minimize(self._coef_loss_function,
                       self.hs,
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
        return loss

    def _gso_loss_function(self, W):
        """
        The loss function for learning the GSO using the pre-trained graph filters. Corresponds to equation 12
        in https://arxiv.org/abs/2003.05729

        Args:
            W (np.array): Learnable weights for graph shift operator (NxN)

        Returns:
            float: value of loss at given iteration 
        """
        # update weights
        self.W = W.reshape(self.N, self.N)
        
        # calculate MSE for each time
        first_graph_filter = self.get_approximate_gso()
        MSE_term = np.linalg.norm(first_graph_filter-self.W, ord=2, axis=(0,1))**2
        loss = .5 * MSE_term
        
        # add in the sparsity term for the gso
        l1_norms = np.linalg.norm(self.W, ord=1, axis=(0,1))
        l1_loss = self.mu[0] * l1_norms
        loss += l1_loss
        
        # add in the commutivity term
        arr = np.dot(self.beta, self.beta).reshape(self.P, self.P, self.N, self.N)
        comm_terms = []
        for i in range(1, self.P):
            comm_terms.append(np.linalg.norm(self.W - self.beta[i], 'fro')**2)
        loss += self.gamma * np.sum(comm_terms)
        return loss

    def predict_with_filter_coefs(self, X):
        """predict the next time-step using learnt GSO and filter coefficients

        Args:
            X (np.array): input features
        Returns:
            np.array: vector of outputs for given inputs 
        """
        # calculate the graph filtered data
        filtered_terms = []
        for i in range(1, self.P+1):
            pgf = PolynomialGraphFilter(self.W, i)
            filtered_terms.append(pgf.filt(X[i-1].T).T)
        term_stack = np.concatenate(filtered_terms, axis=2)

        # weight terms using coefficients and calculate sum
        weighted_terms = np.reshape(self.hs, (1, 1, -1)) * term_stack
        predictions = np.sum(weighted_terms, axis=2)
        return predictions  # T x N

    def _coef_loss_function(self, hs):
        """
        The loss function for learning the graph filter coefficients using the learnt GSO. 
        Corresponds to equation 13 in https://arxiv.org/abs/2003.05729

        Args:
            hs (np.array): Learnable weights for graph filter coefficients (P+1)

        Returns:
            float: value of loss at given iteration 
        """
        # update weights
        self.hs = hs.reshape(self.M)

        # calculate MSE for each time
        MSE_term = np.linalg.norm(self.y-self.predict_with_filter_coefs(self.X), ord=2, axis=1)**2

        # sum the MSE over all time-steps
        powers = np.array([self.y.shape[0] - i for i in range(0, self.y.shape[0])])
        alphas = np.array([self.alpha] * self.y.shape[0])
        powers = np.power(alphas, powers)
        weighted_mses = np.multiply(powers, MSE_term)
        loss = .5 * np.sum(weighted_mses)

        # add in the sparsity term for the parameters
        l1_norms = np.linalg.norm(self.hs, ord=1)
        l1_loss = self.zeta * l1_norms
        loss += l1_loss
        return loss


class AdaptiveGraphAR:
    
    def __init__(self, P, N, alpha, mus, gamma, stepsize_filter, stepsize_weight_matrix, stepsize_debiasing, time_switch_algorithms):
        """
        Adaptive vertex-time graph auto-regressive model, as seen in, https://arxiv.org/abs/2003.05729.
        Here the graph shift operator and the graph filter coefficients are learnt from data.

        Args:
            P (int): Number of auto-regressive terms.
            N (int): Number of vertices of graph / sensor channels.
            alpha (float): Sample weighting / forgetting factor for each time-step [0, 1].
            mus (np.array): Vector of l1 regularisation strengths (P).
            gamma (float): Regularisation strength for commutivity term.
            stepsize_filter (float): Step-size for gradient based projection updates of filter learning.
            stepsize_weight_matrix (float): Step-size for gradient based projection updates of weight matrix learning.
            stepsize_debiasing (float): Step-size for gradient based projection update of debiasing W and h learning.
            time_switch_algorithms (int): Time-step to switch from algorithm 1 (learning) to algorithm 2 (debiasing)
        """
        # initialise attributes
        self.P = P  # number of filter terms
        self.N = N  # number of vertices of graph
        self.alpha = alpha  # regularisation
        self.mus = mus  # regularisation for sparsity
        self.gamma = gamma  # regularisation for commutativity
        self.stepsize_filter = stepsize_filter  # stepsize for learning filter
        self.stepsize_weight_matrix = stepsize_weight_matrix  # stepsize for learning graph topology
        self.stepsize_debiasing = stepsize_debiasing  # stepsize for debiasing algorithm
        self.time_switch_algorithms = time_switch_algorithms
        
        # initialise learnable parameters
        self.beta_pos = np.zeros(shape=(N, N*P))
        self.beta_neg = np.zeros(shape=(N, N*P))
        self.p_matrix = np.zeros(shape=(N, N*P))
        self.r_matrix = np.zeros(shape=(N*P, N*P))
        self.W = np.zeros(shape=(N, N))
        self.W_pos = np.zeros(shape=(N, N))
        self.W_neg = np.zeros(shape=(N, N))

        # matrix of stepsizes
        self.A = self.stepsize_filter * np.eye(P)
        self.stepsize_b = self.stepsize_weight_matrix
        self.stepsize_rho = self.stepsize_debiasing
        
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
        self.identify_graph_topology(X, y)
        self.debias_graph_topology(X, y)

    def identify_graph_topology(self, X, y):
        """
        Learn the graph filters and identify the topology of graph using algorithm 1 
        from https://arxiv.org/abs/2003.05729, by a series of gradient projection 
        updates given in the paper's section 4.

        Args:
            X (np.array): input auto-regressive features for the model
            y (np.array): output / training labels over time.
        """
        filters = []
        weight_matrices = []
        for t in tqdm(range(0, self.time_switch_algorithms)):
            
            # solving for the graph filter
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
            
                # compute G
            self.G = np.matmul(self.beta, self.r_matrix) - (self.p_matrix - self.gamma * self.Q)  # NxNP
            
                # calculate A
            self.beta_pos = self.beta_pos - np.matmul((self.M+self.G), np.kron(self.A, np.eye(self.N)))
            self.beta_neg = self.beta_neg - np.matmul((self.M-self.G), np.kron(self.A, np.eye(self.N)))
            self.beta_pos[self.beta_pos < 0] = 0
            self.beta_neg[self.beta_neg < 0] = 0
            self.beta = np.subtract(self.beta_pos, self.beta_neg)
            filters.append(self.beta)
            
            # estimating the weight matrix
                # compute S
            self.S = np.zeros(shape=(self.N, self.N))
            for k in range(1, self.P):
                filter_k_t = self.beta[:, k*self.N:(k+1)*self.N]
                commute_w_beta = self.commute_loss(self.W, filter_k_t)
                self.S += np.matmul(commute_w_beta, filter_k_t.T) - np.matmul(filter_k_t.T, commute_w_beta)

                # compute V
            first_graph_filter = self.beta[:, :self.N]
            self.V = self.W - (first_graph_filter - self.gamma * self.S)

                # compute weight matrix
            self.W_pos = self.W_pos - self.stepsize_b * (self.mus[0] * np.eye(self.N, self.N) + self.V)
            self.W_pos[self.W_pos < 0] = 0  # keep positive elements only
            self.W_neg = self.W_neg - self.stepsize_b * (self.mus[0] * np.eye(self.N, self.N) - self.V)
            self.W_neg[self.W_neg < 0] = 0  # keep positive elements only
            self.W = np.subtract(self.W_pos, self.W_neg)
            weight_matrices.append(self.W)

        filters = np.stack(filters)  # T x N x NP
        all_weight_matrices = np.stack(weight_matrices)  # TxNxN
        self.filters = filters
        self.all_weight_matrices = all_weight_matrices
        
    def debias_graph_topology(self, X, y):
        """
        Debias the learnt graph topology and identify filter coefficients using algorithm 1 
        from https://arxiv.org/abs/2003.05729, by a series of gradient projection 
        updates given in the paper's section 4.

        Args:
            X (np.array): input auto-regressive features for the model
            y (np.array): output / training labels over time.
        """
        # debias learnt graph topology for the remaining epochs available
        for t in tqdm(range(self.time_switch_algorithms, X.shape[1])):

            # recover weight matrix
            xPt = X[:, t, :].flatten()
            xPt = xPt.reshape(xPt.shape[0], 1)            
            self.r_matrix = self.alpha * self.r_matrix + np.dot(xPt, xPt.T)
            yt = y[t, :]
            yt = yt.reshape(yt.shape[0], 1)
            self.p_matrix = self.alpha * self.p_matrix + np.dot(yt, xPt.T)
            
            # only optimise the non-zero elements of filter (beta) and W
            self.G = np.matmul(self.beta, self.r_matrix) - self.p_matrix
            
            # make elements of G zero where filter is zero and power of W are zero
            g_mask = (self.beta == 0)
            for k in range(0, self.P):
                filter_zeros = g_mask[:, k*self.N:(k+1)*self.N]
                gso = GraphShiftOperator(self.W)
                zeros_w = (gso.power(k+1) == 0)
                g_mask[:, k*self.N:(k+1)*self.N] = filter_zeros * zeros_w
            self.G[g_mask] = 0  # mask G values
            
            # update filter and estimate weights
            self.beta = self.beta - np.matmul(self.G, np.kron(self.A, np.eye(self.N, self.N)))
            self.W = self.beta[:, :self.N]  # approximate W as debiased first graph filter


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


class CausalGraphProcess(GraphAR):
    """
    Causal Graph Process (CGP) as introduced in the paper of Signal Processing on Graphs:
    Causal Modeling of Unstructured Data by Mei and Moura 2017: 
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7763882
    """
    def __init__(self, X, y, N, P, alpha, mu, zeta, gamma=None, init_type='rand', skip_step_2=True):
        super().__init__(X, y, N, P, alpha, mu, zeta, gamma, init_type)
        self.skip_step_2 = skip_step_2
    
    def fit(self, method='BFGS', max_iter=1):
        """ learn parameters of the graph auto-regression model

        Args:
            method (str, optional): Optimizer to use. Defaults to 'BFGS'.
            max_iter (int, optional): Max iterations to use. Defaults to 1.
        """
        print('(1/3) Learning graph filters from data...')
        self._learn_filter(method, max_iter)
        print('(2/3) Learning graph shift operator from graph filters...')
        if self.skip_step_2:
            self.W = self.beta[0]
        else:
            self._learn_gso(method, max_iter)
        print('(3/3) Learning graph filter coefficients from data...')
        self._learn_filter_coefs(method, max_iter)
        
    def _learn_filter(self, method, max_iter):
        """learn filters

        Args:
            method (str): optimisation method to use. 
            max_iter (int): Max iterations to use.
        """
        if self.add_commutivity_term:
            # adding commutivity term makes problem multi-convex
            # learn filters one-by-one using block coordinate descent
            for filter_number in range(0, self.beta.shape[0]):
                print('Learning filter: %s/%s' % (filter_number+1, self.beta.shape[0]))
                res = minimize(self._filter_loss_function,
                            self.beta.flatten(),
                            args=(filter_number),
                            method=method,
                            options={'maxiter': max_iter})
            print('New sweep for filter 1/3...')
            filter_number = 0  # update first learnt filter using knowledge from others
            res = minimize(self._filter_loss_function,
                    self.beta.flatten(),
                    args=(filter_number),
                    method=method,
                    options={'maxiter': max_iter})
        else:
            # problem is convex without the commutivity term
            filter_number = None
            res = minimize(self._filter_loss_function,
                        self.beta.flatten(),
                        args=(filter_number),
                        method=method,
                        options={'maxiter': max_iter})

    def _filter_loss_function(self, beta, filter_number):
        """
        The loss function for the CGP. Corresponds to equation 9
        in https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7763882

        Args:
            beta (np.array): Learnable weights for each graph filter

        Returns:
            float: value of loss at given iteration 
        """
        if filter_number is None:
            # update all weights
            self.beta = beta.reshape(self.P, self.N, self.N)
        else:
            # block coordinate descent: update weights of current filter while keeping others fixed
            self.beta[filter_number] = beta.reshape(self.P, self.N, self.N)[filter_number]
        
        # calculate MSE for each time
        MSE_term = np.linalg.norm(self.y-self.predict(self.X), ord=2, axis=1)**2

        # sum the MSE over all time-steps
        powers = np.array([self.y.shape[0] - i for i in range(0, self.y.shape[0])])
        alphas = np.array([self.alpha] * self.y.shape[0])
        powers = np.power(alphas, powers)
        weighted_mses = np.multiply(powers, MSE_term)
        loss = .5 * np.sum(weighted_mses)

        # add in the sparsity term for the parameters
        l1_norm = np.linalg.norm(self.beta[0], ord=1)
        l1_loss = self.mu * l1_norm
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
    
    def _gso_loss_function(self, W):
        """
        The loss function for learning the GSO using the pre-trained graph filters. Corresponds to equation 12
        in https://arxiv.org/abs/2003.05729

        Args:
            W (np.array): Learnable weights for graph shift operator (NxN)

        Returns:
            float: value of loss at given iteration 
        """
        # update weights
        self.W = W.reshape(self.N, self.N)
        
        # calculate MSE for each time
        first_graph_filter = self.get_approximate_gso()
        MSE_term = np.linalg.norm(first_graph_filter-self.W, ord=2, axis=(0,1))**2
        loss = .5 * MSE_term
        
        # add in the sparsity term for the gso
        l1_norms = np.linalg.norm(self.W, ord=1, axis=(0,1))
        l1_loss = self.mu * l1_norms
        loss += l1_loss
        
        # add in the commutivity term
        arr = np.dot(self.beta, self.beta).reshape(self.P, self.P, self.N, self.N)
        comm_terms = []
        for i in range(1, self.P):
            comm_terms.append(np.linalg.norm(self.W - self.beta[i], 'fro')**2)
        loss += self.gamma * np.sum(comm_terms)
        return loss