from tqdm import tqdm
import numpy as np


def pcn(log_likelihood, u0, y, K, G, n_iters, beta):
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

        u_new = np.sqrt(1 - beta**2) * u_prev + beta * Kc @ np.random.randn(len(u0)) # TODO: Propose new sample using pCN proposal

        ll_new = log_likelihood(u_new, y, G)

        log_alpha = min(ll_new - ll_prev, 0) # TODO: Calculate pCN acceptance probability
        log_u = np.log(np.random.random())

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