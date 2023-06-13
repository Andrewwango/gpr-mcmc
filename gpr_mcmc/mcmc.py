from typing import Callable
from tqdm import tqdm
import numpy as np

log2pi = np.log(2*np.pi)

def log_gaussian_prior(u: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
    """Prior probability according to Gaussian Process zero-mean prior

    Args:
        u (np.ndarray): data
        K_inv (np.ndarray): inverse covariance matrix

    Returns:
        np.ndarray: prior probability of data
    """
    return -0.5 * (len(u) * log2pi - np.linalg.slogdet(K_inv)[1] + u @ K_inv @ u)

def cholesky_decomposition(K: np.ndarray) -> np.ndarray:
    """Calculate Cholesky decomposition of covariance matrix K

    Args:
        K (np.ndarray): covariance matrix

    Returns:
        np.ndarray: Cholesky decomposition C such that K = C @ C.T
    """
    N = K.shape[0]
    return np.linalg.cholesky(K + 1e-6 * np.eye(N))

def mcmc_mh_grw(
        log_likelihood: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], 
        data: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray,
        n_iters: int,
        beta: float,
        rng: np.random.Generator = np.random.default_rng(),
        log_prior: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
        ):
    """MCMC using Gaussian random walk proposal to sample from log_likelihood, assuming the Gaussian Process prior.

    Args:
        log_likelihood (Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]): log likelihood to sample from.
        data (np.ndarray): observed data
        K (np.ndarray): Gaussian process prior covariance matrix
        mask (np.ndarray): data observation mask
        n_iters (int): number of iterations to run MCMC
        beta (float): step-size
        rng (np.random.Generator, optional): random number generator. Defaults to np.random.default_rng().
        log_prior (Callable[[np.ndarray, np.ndarray], np.ndarray] | None, optional): log prior distribution, if None this is the Gaussian Process prior. Defaults to None.

    Returns:
        X: list of samples
    """
    
    log_prior = log_prior if log_prior is not None else log_gaussian_prior

    # Note in GRW we are actually sampling from the posterior "target", incorporating the Gaussian process prior
    def log_posterior(u, v, K_inverse, mask):
        return log_prior(u, K_inverse) + log_likelihood(u, v, mask)

    # Compute inverse
    Kc = cholesky_decomposition(K)
    Kc_inv = np.linalg.inv(Kc)
    K_inv = Kc_inv @ Kc_inv.T

    X = []

    N = K.shape[0]
    u_prev = Kc @ rng.standard_normal(N)
    posterior_prev = log_posterior(u_prev, data, K_inv, mask)

    for _ in tqdm(range(n_iters)):
        # Proposal step
        u = u_prev + beta * Kc @ rng.standard_normal(N)

        # Calculate posterior with this sample
        posterior = log_posterior(u, data, K_inv, mask)

        # Metropolis-Hastings accept/reject
        accept = np.log(rng.random()) <= min(posterior - posterior_prev, 0)
        if accept:
            X.append(u)
            u_prev = u
            posterior_prev = posterior
        else:
            X.append(u_prev)

    return X

def mcmc_mh_pcn(
        log_likelihood: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], 
        data: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray,
        n_iters: int,
        beta: float,
        rng: np.random.Generator = np.random.default_rng()        
        ):
    """MCMC using pCN proposal to sample from log_likelihood.

    Args:
        log_likelihood (Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]): log likelihood to sample from.
        data (np.ndarray): observed data
        K (np.ndarray): Gaussian process prior covariance matrix
        mask (np.ndarray): data observation mask
        n_iters (int): number of iterations to run MCMC
        beta (float): step-size
        rng (np.random.Generator, optional): random number generator. Defaults to np.random.default_rng().

    Returns:
        X: list of samples
    """

    # Cholesky decomposition
    Kc = cholesky_decomposition(K)

    X = []

    N = K.shape[0]
    u_prev = Kc @ rng.standard_normal(N)
    likelihood_prev = log_likelihood(u_prev, data, mask)

    for _ in tqdm(range(n_iters)):
        # pCN proposal step
        u = np.sqrt(1 - beta**2) * u_prev + beta * Kc @ rng.standard_normal(N)

        # Calculate likelihood. Note that prior isn't used here as it cancels out in the acceptance step
        likelihood = log_likelihood(u, data, mask)

        # Metropolis-Hastings accept/reject
        accept = np.log(rng.random()) <= min(likelihood - likelihood_prev, 0)
        if accept:
            X.append(u)
            u_prev = u
            likelihood_prev = likelihood
        else:
            X.append(u_prev)

    return X