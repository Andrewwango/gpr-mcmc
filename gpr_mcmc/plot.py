import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import get_cmap
from copy import copy

def plot_2D(ax: Axes, data: np.ndarray, xi: np.ndarray, yi: np.ndarray, title: str = None, pos_only: bool = True):
    """Plot 2D data onto x,y coordinates. Only supports integer coordinates

    Args:
        ax (Axes): matplolib axes to plot onto
        data (np.ndarray): data to plot
        xi (np.ndarray): x coordinates of data (integers only)
        yi (np.ndarray): y coordinates of data (integers only)
        title (str, optional): axis title. Defaults to None.
        pos_only (bool, optional): whether data is constrained to be positive. Defaults to True.
    """
    
    Z = -np.ones((max(yi) + 1, max(xi) + 1))
    for i in range(len(data)):
        Z[(yi[i], xi[i])] = data[i]
    
    my_cmap = copy(get_cmap('viridis'))
    my_cmap.set_under('k', alpha=0)
    
    im = ax.imshow(Z, origin='lower', cmap=my_cmap, clim=[-0.1 if pos_only else np.min(data), np.max(data)])
    
    if title is not None:
        ax.set_title(title)

    plt.colorbar(im)