import numpy as np
from scipy.spatial import distance_matrix

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn import clone

from .mcmc import pcn
from .testgrid import TestGrid
from .likelihoods import *

def expected_counts_from_samples(samples, latent_mapper=None, c_squared=False):
    # c_squared just convenience parameter for easily calculating variance
    us = np.array(samples) if latent_mapper is None else latent_mapper(np.array(samples))
    return np.mean(us if not c_squared else us + us**2, axis=0)

def variance_from_samples(samples, latent_mapper=None):
    return expected_counts_from_samples(samples, c_squared=True, latent_mapper=latent_mapper)\
        - expected_counts_from_samples(samples, latent_mapper=latent_mapper)**2

def GaussianKernel(x, l):
    """ Generate Gaussian kernel matrix efficiently using scipy's distance matrix function"""
    # TODO: test generic sklearn kernel and remove this
    D = distance_matrix(x, x)
    return np.exp(-pow(D, 2)/(2*pow(l, 2)))

class GaussianProcessMCMCRegressor:
    def __init__(self, kernel=None, likelihood='poisson', optimizer=None, random_state=None, latent_mapper=None,
                 n=10000, beta=0.2): 
        #docs for likelihood: describes the log likelihood of the underlying latent field u \in \mathbb{R} to be sampled.
        #if passing a custom likelihood, and your latent field is not \mathbb{R} (e.g. Poisson must be positive only) then
        #also set latent_mapper to describe the auxiliary variable mapping (e.g. np.exp for Poisson)

        self.kernel = kernel
        #TODO: build kernel API using sklearn kernels, default is RBF kernel like below with ell=2
        self.likelihood = likelihood
        #TODO: if optimizer is passed, perform scipy.optimize.minimize on log_marginal_likelihood with kernel parameters 
        self.optimizer = optimizer
        #TODO: put random states in all random funcs e.g. MCMC
        self.random_state = random_state
        self.latent_mapper = latent_mapper

        ### Set the likelihood and target, for sampling p(u|c)
        if isinstance(self.likelihood, str):
            if latent_mapper is not None:
                raise ValueError("latent_mapper must not be passed when using builtin likelihood functions")
            if self.likelihood == 'poisson':
                self.log_likelihood = log_poisson_likelihood
                self.latent_mapper = np.exp
            elif self.likelihood in ('gaussian', 'normal'):
                self.log_likelihood = log_normal_likelihood
                #TODO: warnings warn that this will be slower than closed form sklearn.gaussian_process.GaussianProcessRegressor
            elif self.likelihood in ('probit', 'classification'):
                self.log_likelihood = log_probit_likelihood
                #TODO: check if probit has a latent mapper
            else:
                raise ValueError("likelihood must be one of poisson, gaussian or probit")
        elif callable(self.likelihood):
            self.log_likelihood = self.likelihood
        else:
            raise ValueError("likelihood must be a str name or a callable")
        
        self.n = n
        self.beta = beta
        self.ell = 2

        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = ConstantKernel(1.0, constant_value_bounds="fixed") \
                * RBF(self.ell, length_scale_bounds="fixed")
        else:
            self.kernel_ = clone(self.kernel)

    #TODO: what is best practice for fit+predict methods?
    def fit_and_predict(self, X, y, X_test, G=None):
        if type(X_test) is TestGrid:
            X_test = X_test.X
        M = X.shape[0]
        N = X_test.shape[0] + M
        X_dash = np.vstack([X_test, X])
        #X_dash = X_test

        G = np.hstack([np.zeros((M, N - M)), np.eye(M)])

        K = self.kernel_(X_dash)
        Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
        u0 = Kc @ np.random.randn(N, )

        samples, acc_pcn = pcn(self.log_likelihood, u0, y, K, G, self.n, self.beta)

        y_dash_pred_bar = expected_counts_from_samples(samples, latent_mapper=self.latent_mapper)
        y_dash_pred_var = variance_from_samples(samples, latent_mapper=self.latent_mapper)

        y_pred_bar = y_dash_pred_bar[:N-M]
        y_pred_var = y_dash_pred_var[:N-M]

        return y_pred_bar
    
    def score(self, y, y_pred):
        error_field = np.abs(y - y_pred)
        mae = np.mean(error_field)
        return mae
    
    def log_marginal_likelihood(self):
        pass

    
