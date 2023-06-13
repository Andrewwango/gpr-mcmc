from tqdm import tqdm
import numpy as np

log2pi = np.log(2*np.pi)

def log_gaussian_prior(u, K_inverse):
    return -0.5 * (len(u)*log2pi - np.linalg.slogdet(K_inverse)[1] + u @ K_inverse @ u)

def mcmc_mh_grw(log_likelihood, u0, data, K, G, n_iters, beta, rng=np.random.default_rng(), log_prior=None):
    """ Gaussian random walk Metropolis-Hastings MCMC method
        for sampling from pdf defined by log_target.
    Inputs:
        log_target - log-target density
        u0 - initial sample
        y - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""
    
    log_prior = log_prior if log_prior is not None else log_gaussian_prior

    def log_target(u, y, K_inverse, G):
        return log_prior(u, K_inverse) + log_likelihood(u, y, G)

    X = []
    acc = 0
    u_prev = u0

    # Inverse computed before the for loop for speed
    N = K.shape[0]
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    Kc_inverse = np.linalg.inv(Kc)
    K_inverse = Kc_inverse @ Kc_inverse.T # TODO: compute the inverse of K using its Cholesky decomopsition

    lt_prev = log_target(u_prev, data, K_inverse, G)

    for i in tqdm(range(n_iters)):

        u_new = u_prev + beta * Kc @ rng.standard_normal(len(u0)) # TODO: Propose new sample - use prior covariance, scaled by beta

        lt_new = log_target(u_new, data, K_inverse, G)

        log_alpha = min(lt_new - lt_prev, 0) # TODO: Calculate acceptance probability based on lt_prev, lt_new
        log_u = np.log(rng.random())

        # Accept/Reject
        accept = log_u <= log_alpha # TODO: Compare log_alpha and log_u to accept/reject sample (accept should be boolean)
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            lt_prev = lt_new
        else:
            X.append(u_prev)

    return X, acc / n_iters

def mcmc_mh_pcn(log_likelihood, u0, y, K, G, n_iters, beta, rng=np.random.default_rng()):
    """ pCN MCMC method for sampling from pdf defined by log_prior and log_likelihood.
    Inputs:
        log_likelihood - log-likelihood function
        u0 - initial sample
        y - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""
    N = K.shape[0]
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    X = []
    acc = 0
    u_prev = u0

    ll_prev = log_likelihood(u_prev, y, G)

    for i in tqdm(range(n_iters)):

        u_new = np.sqrt(1 - beta**2) * u_prev + beta * Kc @ rng.standard_normal(len(u0)) # TODO: Propose new sample using pCN proposal

        ll_new = log_likelihood(u_new, y, G)

        log_alpha = min(ll_new - ll_prev, 0) # TODO: Calculate pCN acceptance probability
        log_u = np.log(rng.random())

        # Accept/Reject
        accept = log_u <= log_alpha # TODO: Compare log_alpha and log_u to accept/reject sample (accept should be boolean)
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            ll_prev = ll_new
        else:
            X.append(u_prev)

    return X, acc / n_iters