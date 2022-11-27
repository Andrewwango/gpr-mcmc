import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

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

class TestGrid:
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
    
    def X(self):
        return self.X
    
    @staticmethod
    def plot(X, y):
        if type(X) is TestGrid:
            X = X.X
        plot_2D(y, X[:,0], X[:,1], title=f"Counts")