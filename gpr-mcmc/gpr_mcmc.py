import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import norm
from scipy.special import factorial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from .mcmc import pcn
from .testgrid import TestGrid

def expected_counts_from_samples(samples, c_squared=False):
    us = np.array(samples)
    theta = np.exp(us)
    return np.mean(theta if not c_squared else theta + theta**2, axis=0)

def variance_from_samples(samples):
    return expected_counts_from_samples(samples, c_squared=True) - expected_counts_from_samples(samples)**2

class GaussianProcessMCMCRegressor:
    def __init__(self, kernel=None, likelihood='poisson', optimizer=None, random_state=None): 
        
        self.kernel = kernel
        #TODO: build kernel API using sklearn kernels, default is RBF kernel like below with ell=2
        self.likelihood = likelihood
        #TODO: if optimizer is passed, perform scipy.optimize.minimize on log_marginal_likelihood with kernel parameters 
        self.optimizer = optimizer
        #TODO: put random states in all random funcs e.g. MCMC
        self.random_state = random_state

        ### Set the likelihood and target, for sampling p(u|c)
        if self.likelihood == 'poisson':
            self.log_likelihood = self.log_poisson_likelihood
        elif self.likelihood == 'gaussian':
            #TODO: if likelihood is gaussian, redirect all calls to sklearn.gaussian_process.GaussianProcessRegressor
            raise ...
        
        self.n = 10000
        self.beta = 0.2
        self.ell = 2

        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = ConstantKernel(1.0, constant_value_bounds="fixed") \
                * RBF(self.ell, length_scale_bounds="fixed")
        else:
            self.kernel_ = clone(self.kernel)

    def log_poisson_likelihood(self, u, c, G):
        Gu = G @ u
        return np.sum(c * Gu - np.exp(Gu) - np.log(factorial(c, exact=False)))

    def GaussianKernel(self, x, l):
        """ Generate Gaussian kernel matrix efficiently using scipy's distance matrix function"""
        D = distance_matrix(x, x)
        return np.exp(-pow(D, 2)/(2*pow(l, 2)))

    #TODO: what is best practice for fit+predict methods?
    def fit_and_predict(self, X, y, X_test, G=None):
        if type(X_test) is TestGrid:
            X_test = X_test.X
        M = X.shape[0]
        N = X_test.shape[0] + M
        X_dash = np.vstack([X_test, X])
        #X_dash = X_test

        G = np.hstack([np.zeros((M, N - M)), np.eye(M)])

        K = self.GaussianKernel(X_dash, self.ell)
        Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
        u0 = Kc @ np.random.randn(N, )

        samples, acc_pcn = pcn(self.log_likelihood, u0, y, K, G, self.n, self.beta)

        y_dash_pred_bar = expected_counts_from_samples(samples)
        y_dash_pred_var = variance_from_samples(samples)

        y_pred_bar = y_dash_pred_bar[:N-M]
        y_pred_var = y_dash_pred_var[:N-M]

        return y_pred_bar
    
    def score(self, y, y_pred):
        error_field = np.abs(y - y_pred)
        mae = np.mean(error_field)
        return mae
    
    def log_marginal_likelihood(self):
        pass

    
