"""
Note, when implementing a custom log_likelihood function you may need to create an auxiliary variable
if the domain of the function is restricted, e.g. for Poisson distributions (positive x) or for 
classification distributions (0<x<1). In these cases you will need to create a function that matches 
the auxiliary transformation to recover original samples from the auxiliary function. See 
log_poisson_likelihood and log_probit_likelihood for examples.
"""

import numpy as np
from scipy.special import factorial
from scipy.stats import norm

log2pi = np.log(2*np.pi)

def log_normal_likelihood(u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Log Gaussian likelihood (assumes constant variance across entire field).
    If your data is modelled by a Gaussian likelihood, the solution is available
    in closed form so use instead 
    [`sklearn.gaussian_process.GaussianProcessRegressor`](https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-regression-gpr).

    Args:
        u (np.ndarray): underlying field
        v (np.ndarray): observed data
        mask (np.ndarray): data observation mask

    Returns:
        np.ndarray: observation likelihood p(v|u)
    """
    mu = v - mask @ u
    return -0.5 * (len(v) * log2pi + mu @ mu)


def log_probit_likelihood(u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Log Probit likelihood. This is the Gaussian CDF which can be used to model
    binary classification data. Note that latent_mapper must be set to norm.cdf. 
    This is because the domain is restricted to [0,1].

    Args:
        u (np.ndarray): underlying field
        v (np.ndarray): observed data
        mask (np.ndarray): data observation mask

    Returns:
        np.ndarray: observation likelihood p(v|u)
    """
    phi = norm.cdf(mask @ u)
    return v @ np.log(phi) + (1. - v) @ np.log(1. - phi)


def log_poisson_likelihood(u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Log Poisson likelihood. Note that latent_mapper must be set to np.exp. 
    This is because a Poisson field must be positive, so we set an auxiliary variable 
    to equal np.exp(u) to infer u as usual.

    Args:
        u (np.ndarray): underlying field
        v (np.ndarray): observed data
        mask (np.ndarray): data observation mask

    Returns:
        np.ndarray: observation likelihood p(v|u)
    """
    Gu = mask @ u
    return np.sum(v * Gu - np.exp(Gu) - np.log(factorial(v, exact=False)))

