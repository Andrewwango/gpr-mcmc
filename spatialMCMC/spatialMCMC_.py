import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import norm
from scipy.special import factorial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
from .MCMC import pcn

def plot_2D(counts, xi, yi, title=None, colors='viridis'):
    """Visualise count data given the index lists"""
    Z = -np.ones((max(yi) + 1, max(xi) + 1))
    for i in range(len(counts)):
        Z[(yi[i], xi[i])] = counts[i]
    my_cmap = copy.copy(cm.get_cmap(colors))
    my_cmap.set_under('k', alpha=0)
    fig, ax = plt.subplots()
    im = ax.imshow(Z, origin='lower', cmap=my_cmap, clim=[-0.1, np.max(counts)])
    fig.colorbar(im)
    if title:  plt.title(title)
    plt.show()

def expected_counts_from_samples(samples, c_squared=False):
    us = np.array(samples)
    theta = np.exp(us)
    return np.mean(theta if not c_squared else theta + theta**2, axis=0)

def variance_from_samples(samples):
    return expected_counts_from_samples(samples, c_squared=True) - expected_counts_from_samples(samples)**2

class SpatialRegression:
    def __init__(self, likelihood='poisson'): 
        self.n = 10000
        self.beta = 0.2
        self.ell = 2
        self.likelihood = likelihood

        ### Set the likelihood and target, for sampling p(u|c)
        self.log_likelihood = self.log_poisson_likelihood

    def log_poisson_likelihood(self, u, c, G):
        Gu = G @ u
        return np.sum(c * Gu - np.exp(Gu) - np.log(factorial(c, exact=False)))

    def GaussianKernel(self, x, l):
        """ Generate Gaussian kernel matrix efficiently using scipy's distance matrix function"""
        D = distance_matrix(x, x)
        return np.exp(-pow(D, 2)/(2*pow(l, 2)))

    def log_likelihood(self):
        if self.likelihood == 'poisson':
            return self.log_poisson_likelihood

    def fit_and_predict(self, X, y, X_test, G=None):
        if type(X_test) is DataGrid:
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
    
    def plot(self, X, y):
        return plot_2D(y, X[:,0], X[:,1], title=f"Counts")

class DataGrid:
    def __init__(self, coords=None, grid_limits=None, grid_resolutions=None, grid_ns=None):
        coords_args_on = coords is not None
        grid_args_on = grid_limits is not None or grid_resolutions is not None or grid_ns is not None
        other_args_on = False
        if coords_args_on and not grid_args_on and not other_args_on:
            self.X = self._grid_from_coords(coords)
        elif not coords_args_on and grid_args_on and not other_args_on:
            self.X = self._grid_from_limits(grid_limits, grid_resolutions, grid_ns)
        elif not coords_args_on and not grid_args_on and other_args_on:
            self.X = self._grid_from_other()
        else:
            self.X = None
            raise ValueError("Bad combination of TestGrid inputs")
       
    def _grid_from_coords(self, coords):
        return coords
    
    def _grid_from_limits(self, grid_limits, grid_resolutions, grid_ns):
        if grid_limits is not None:
            dims = len(grid_limits)
            
        if grid_limits is not None and grid_ns is not None:
            axes = [np.linspace(*grid_limits[i], num=grid_ns[i], endpoint=False) for i in range(dims)]
        elif grid_limits is not None and grid_resolutions is not None:
            axes = [np.arange(*grid_limits[i], step=grid_resolutions[i]) for i in range(dims)]
        else:
            raise ValueError("Only combinations of limits and resolutions or limits and ns is allowed")
        
        return np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, dims)
    
    def _grid_from_other(self):
        pass
    
    @staticmethod
    def plot(X, y):
        if type(X) is DataGrid:
            X = X.X
        plot_2D(y, X[:,0], X[:,1], title=f"Counts")