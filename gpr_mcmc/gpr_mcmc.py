import warnings
from typing import Callable
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel
from sklearn.metrics import r2_score
from sklearn import clone

from .mcmc import mcmc_mh_grw, mcmc_mh_pcn
from .plot import plot_2D
from .likelihoods import *

def expected_predpost_from_samples(samples, latent_mapper=None, c_squared=False):
    # c_squared just convenience parameter for easily calculating variance
    us = np.array(samples) if latent_mapper is None else latent_mapper(np.array(samples))
    return np.mean(us if not c_squared else us + us**2, axis=0)

def variance_predpost_from_samples(samples, latent_mapper=None):
    return expected_predpost_from_samples(samples, c_squared=True, latent_mapper=latent_mapper)\
        - expected_predpost_from_samples(samples, latent_mapper=latent_mapper)**2

def check_random_state(random_state):
    # Turn random_state seed into a np.random.Generator instance. Also accepts existing generators.
    # This bypasses sklearn.utils.check_random_state, which does not yet accept random generators.
    if isinstance(random_state, np.random.Generator):
        return random_state
    elif isinstance(random_state, int) or random_state is None:
        return np.random.default_rng(random_state)
    else: #(including isinstance(random_state, np.random.RandomState))
        raise ValueError("Random state must be passed as integer or np.random.Generator")

def check_likelihood(log_likelihood, latent_mapper):
    if isinstance(log_likelihood, str):
        if latent_mapper is not None:
            raise ValueError("latent_mapper must not be passed when using builtin likelihood functions")

        if log_likelihood == 'poisson':
            log_likelihood = log_poisson_likelihood
            latent_mapper = np.exp
        elif log_likelihood in ('gaussian', 'normal'):
            log_likelihood = log_normal_likelihood
            latent_mapper = None
            warnings.warn("""Warning: for conjugate likelihoods, such as 'gaussian', exact Gaussian Process Regression
            is available in closed form using sklearn.gaussian_process.GaussianProcessRegressor and is therefore
            much faster than a sampling-based approach.""")
        elif log_likelihood in ('probit', 'classification'):
            log_likelihood = log_probit_likelihood
            latent_mapper = norm.cdf
        else:
            raise ValueError("log_likelihood must be one of poisson, gaussian or probit")

    elif callable(log_likelihood):
        log_likelihood = log_likelihood # TODO: check likelihood is a valid function
        raise NotImplementedError("TODO: derive predictive posterior for this likelihood and check if it permits a latent_mapper")
    else:
        raise ValueError("log_likelihood must be a str name or a callable")
    
    return log_likelihood, latent_mapper

class GaussianProcessMCMCRegressor:
    """Gaussian Process regression using robust MCMC sampling.

    Usage:
    ```python
    >>> import numpy as np
    >>> from gpr_mcmc import GaussianProcessMCMCRegressor as GPMR
    >>> gpmr = GPMR(
        ell=0.2,
        log_likelihood='poisson'
        )
    >>> gpmr.fit(X_train, y_train)
    gpr_mcmc.gpr_mcmc.GaussianProcessMCMCRegressor
    >>> gpmr.score(X_test, y_test, method="mae")
    ```

    Args:
        kernel (sklearn.gaussian_process.kernels.Kernel, optional): Any GPR kernel from `scikit-learn` kernels. If None, RBF kernel is used with ell given by `ell`. Defaults to None.
        log_likelihood (str | callable], optional): If str, must be one of `poisson`, `gaussian` or `probit`. If a callable, see `likelihoods.py` for example implementations. See below for more info Defaults to 'poisson'.
        optimizer (None, optional): Not yet implemented. Defaults to None.
        random_state (int | np.random.Generator | None, optional): If int, seeds a new numpy random number generator. Prefer using a Generator as per numpy recommendations. Defaults to None.
        latent_mapper (None | callable, optional): Ignore if using built-in likelihood functions. If using a custom likelihood, see `likelihoods.py` for tips. Defaults to None.
        n (int, optional): Iterations used for MCMC sampling. Defaults to 10000.
        beta (float, optional): step size using in MCMC proposals. Controls the "temperature" of the sampling. Defaults to 0.2.
        ell (float | None, optional): Length scale of the RBF kernel if kernel is None. If None, this defaults to 2. Defaults to None.
        mcmc_method (str, optional): MCMC variant to use. "grw" is Gaussian random walk - Metropolis-Hastings with a Gaussian prior proposal. "pcn" is preconditioned Crank-Nicholson which does not degenerate in infinite-dimensions. Defaults to "pcn".
    
    Note on likelihoods: this describes the log likelihood of the underlying latent field u \in \mathbb{R} to be sampled.
    If passing a custom likelihood, and your latent field is not \mathbb{R} (e.g. Poisson must be positive only) then also 
    set latent_mapper to describe the auxiliary variable mapping (e.g. np.exp for Poisson)
    
    """
    def __init__(
            self, 
            kernel: Kernel = None, 
            log_likelihood: str | Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = 'poisson',
            optimizer: None = None,
            random_state: int | np.random.Generator | None = None, 
            latent_mapper: None | Callable[[np.ndarray], np.ndarray] = None,
            n: int = 10000,
            beta: float = 0.2, 
            ell: float | None = None,
            mcmc_method: str = "pcn"
            ): 

        self.kernel = kernel
        self.log_likelihood, self.latent_mapper = check_likelihood(log_likelihood, latent_mapper)
        self.rng = check_random_state(random_state)
        
        #TODO: if optimizer is passed, perform scipy.optimize.minimize on log_marginal_likelihood with kernel parameters 
        self.optimizer = optimizer
        
        if mcmc_method == "pcn":
            self.mcmc = mcmc_mh_pcn
        elif mcmc_method == "grw":
            self.mcmc = mcmc_mh_grw
        else:
            raise ValueError("mcmc_method must be one of 'pcn' for the preconditioned Crank-Nicholson or 'grw' for Gaussian random walk")

        self.n = n
        self.beta = beta
        self.ell = 2 if ell is None else ell

        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(self.ell, length_scale_bounds="fixed")
        elif self.kernel is not None and ell is not None:
            raise ValueError("ell argument can only be specified when kernel is not specified, in order to use the default RBF kernel with length-scale ell.")
        else:
            self.kernel_ = clone(self.kernel)

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        """Fit Gaussian Process Regression model using MCMC

        Args:
            X (np.ndarray): training data features of shape (n_samples, n_features)
            y (np.ndarray): training data targets of shape (n_samples,)

        Returns:
            object: fitted model
        """
        #TODO: check inputs
        self.X_train_ = X
        self.y_train_ = y

        return self

    def predict(self, X: np.ndarray, return_train_preds: bool = False) -> np.ndarray:
        """Predict using the Gaussian process regression model. 

        Args:
            X (np.ndarray): test data features of shape (n_samples, n_features)
            return_train_preds (bool, optional): whether to test the model on the original training points too. Defaults to False.

        Returns:
            np.ndarray: predictions on test data
        """
        #TODO: check input

        check_is_fitted(self, ("X_train_", "y_train_"))

        M = self.X_train_.shape[0]
        N = X.shape[0] + M
        X_dash = np.vstack([X, self.X_train_]) # all data

        mask = np.hstack([np.zeros((M, N - M)), np.eye(M)])

        K = self.kernel_(X_dash)
        
        samples = self.mcmc(self.log_likelihood, self.y_train_, K, mask, self.n, self.beta, self.rng)

        y_dash_pred_bar = expected_predpost_from_samples(samples, latent_mapper=self.latent_mapper)
        y_dash_pred_var = variance_predpost_from_samples(samples, latent_mapper=self.latent_mapper)

        y_pred_bar = y_dash_pred_bar if return_train_preds else y_dash_pred_bar[:N-M]
        y_pred_var = y_dash_pred_var if return_train_preds else y_dash_pred_var[:N-M]

        return y_pred_bar
    
    def score(self, X: np.ndarray, y: np.ndarray, method: str = "r2") -> float:
        """Calculate score based on test data and test data ground truth targets

        Args:
            X (np.ndarray): test data features of shape (n_samples, n_features)
            y (np.ndarray): test data ground truth targets of shape (n_samples,)
            method (str, optional): R2 (coefficient of determination) or MAE (mean absolute error). Defaults to "r2".

        Returns:
            float: model score
        """
        y_pred = self.predict(X, return_train_preds=False)

        if method in ("r2", "r2_score"):
            func = r2_score
        elif method == "mae":
            func = self.mae
        else:
            raise ValueError("method must be either r2 or mae")

        return func(y, y_pred)

    def mae(self, y, y_pred):
        #TODO: just use metrics.mean_absolute_error
        error_field = np.abs(y - y_pred)
        mae = np.mean(error_field)
        return mae
    
    def log_marginal_likelihood(self):
        #TODO: implement this
        pass

    @staticmethod
    def plot(X: np.ndarray, y: np.ndarray, ax=None, title=None, pos_only=True):
        if ax is None:
            fig, ax = plt.subplots()

        plot_2D(ax, y, X[:,0], X[:,1], title=title, pos_only=pos_only)

    
