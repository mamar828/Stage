import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from src.hdu.arrays.array import Array


class Array2D(Array):
    """
    Encapsulates the methods specific to two-dimensional arrays.
    """

    def plot(self, ax, **kwargs):
        """
        Plots an Array2D onto an axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis on which to plot the Array2D.
        kwargs : dict
            Additional parameters to parametrize the plot. Supported keywords and types are :
            "xlabel" : str, default=None. Specify the label for the x axis.
            "ylabel" : str, default=None. Specify the label for the y axis.
            "xlim" : str, default=None. Specify the x bounds.
            "ylim" : str, default=None. Specify the y bounds.
            "cbar_label" : str, default=None. Specify the label for the colorbar.
            "discrete_colormap" : bool, default=False. Specify if the colormap should be discrete.
            "cbar_limits" : tuple, default=None. Specify the limits of the colorbar. Essential for a discrete_colormap.
        """
        if kwargs.get("discrete_colormap"):
            viridis_cmap = plt.cm.viridis
            cbar_limits = kwargs["cbar_limits"]
            interval = (cbar_limits[1] - cbar_limits[0]) * 2
            bounds = np.linspace(*cbar_limits, interval + 1)
            cmap = ListedColormap(viridis_cmap(np.linspace(0, 1, interval)))
            norm = BoundaryNorm(bounds, cmap.N)
            imshow = ax.imshow(self.data, origin="lower", cmap=cmap, norm=norm)
            cbar = plt.colorbar(imshow, ticks=np.linspace(*cbar_limits, interval//2 + 1), fraction=0.046, pad=0.04)

        else:
            imshow = ax.imshow(self.data, origin="lower")
            cbar = plt.colorbar(imshow, fraction=0.046, pad=0.04)

        if kwargs.get("cbar_limits") and not kwargs.get("discrete_colormap"):
            imshow.set_clim(*kwargs.get("cbar_limits"))
        if kwargs.get("cbar_label"):
            cbar.set_label(kwargs.get("cbar_label"))
        if kwargs.get("xlabel"):
            plt.xlabel(kwargs.get("xlabel"))
        if kwargs.get("ylabel"):
            plt.ylabel(kwargs.get("ylabel"))
        if kwargs.get("xlim"):
            ax.set_xlim(*kwargs.get("xlim"))
        if kwargs.get("ylim"):
            ax.set_ylim(*kwargs.get("ylim"))
        
        ax.tick_params(axis='both', direction='in')
