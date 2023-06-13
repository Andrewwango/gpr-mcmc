# Gaussian Process Regression using Monte Carlo Markov Chain sampling
Gaussian Process Regression (GPR) with non-Gaussian likelihoods using robust infinite-dimension Monte Carlo Markov Chain (MCMC) sampling.

## Why?
Gaussian Process Regression is only available in exact, closed form when the data likelihood is assumed to be Gaussian. However, in many use-cases data can be:

- Discrete count data (such as population), in which case the most appropriate likelihood is Poisson;
- Binary classification data, in which case an appropriate likelihood could be the probit model.

To deal with this, there are a number of approximation methods, such as the Laplace approximation. See [[Murphy: Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) (Section 18.4)] for an excellent treatment. The `scikit-learn` [GP classification](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier) uses the Laplace method.

The MCMC sampling approach is attractive as we can theoretically sample from any non-conjugate likelihood to simulate an exact solution. This works using a Metropolis-Hastings style accept/reject step informed by the given likelihood.

However, the performance of traditional Metropolis-Hastings suffers in high dimensions such as in spatial problems where there are many data points, because the acceptance probability in the Metropolis step degenerates to zero. In this repo we implement MCMC using the [preconditioned Crank-Nicholson proposal step](https://en.wikipedia.org/wiki/Preconditioned_Crank%E2%80%93Nicolson_algorithm), which is well-defined in infinite dimensions.

## Usage

Use the algorithm as you would normally use `scikit-learn` [Gaussian Process regression](sklearn.gaussian_process.GaussianProcessRegressor) or [classification](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier):

```python
>>> import numpy as np
>>> from gpr_mcmc import GaussianProcessMCMCRegressor as GPMR
>>> gpmr = GPMR(
    ell=0.2, # default RBF kernel length
    log_likelihood='poisson',
    random_state=np.random.default_rng(42) # seed for MCMC sampling
    )
>>> gpmr.fit(X_train, y_train)
gpr_mcmc.gpr_mcmc.GaussianProcessMCMCRegressor
>>> gpmr.score(X_test, y_test, method="mae") # mean absolute error
```

See [demo.ipynb](demo.ipynb) for a real example using spatial data.
### Main options

- `mcmc_method`: MCMC variant to use. "grw" is Gaussian random walk - Metropolis-Hastings with a Gaussian prior proposal. "pcn" is preconditioned Crank-Nicholson which does not degenerate in infinite-dimensions.
- `log_likelihood`: If str, must be one of `poisson`, `gaussian` or `probit`. If a callable, see [likelihoods.py](likelihoods.py) for example implementations.
- `kernel`: Any GPR kernel from `scikit-learn` [Kernel API](https://scikit-learn.org/stable/modules/gaussian_process.html#kernels-for-gaussian-processes). If None, RBF kernel is used with ell given by `ell`.
- `beta`: step size using in MCMC proposals. Controls the "temperature" of the sampling.
