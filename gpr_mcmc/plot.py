import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

def plot_2D(ax, counts, xi, yi, title=None, colors='viridis', pos_only=True):
    """Visualise count data given the index lists"""
    Z = -np.ones((max(yi) + 1, max(xi) + 1))
    for i in range(len(counts)):
        Z[(yi[i], xi[i])] = counts[i]
    my_cmap = copy.copy(cm.get_cmap(colors))
    my_cmap.set_under('k', alpha=0)
    
    im = ax.imshow(Z, origin='lower', cmap=my_cmap, clim=[-0.1 if pos_only else np.min(counts), np.max(counts)])
    if title is not None:
        ax.set_title(title)
    plt.colorbar(im)