import numpy as np
from scipy.special import factorial
from scipy.stats import norm

log2pi = np.log(2*np.pi)

def log_continuous_likelihood(u, v, G):
    # Return observation likelihood p(v|u)
    n = len(v)
    mu = v - G @ u
    return -0.5 * (n*log2pi + mu @ mu)


def log_probit_likelihood(u, t, G):
    phi = norm.cdf(G @ u)
    # Return likelihood p(t|u)
    return t @ np.log(phi) + (1. - t) @ np.log(1. - phi)


def log_poisson_likelihood(u, c, G):
    # Return likelihood p(c|u)
    # NOTE: because Poisson field must be positive, we set an auxiliary variable to equal np.exp(u) tp
    # infer u as usual. To recover Poisson samples from u, we set latent_mapper to np.exp
    Gu = G @ u
    return np.sum(c * Gu - np.exp(Gu) - np.log(factorial(c, exact=False)))


def log_normal_likelihood():
    #TODO: implement gaussian likelihood
    pass