import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.animation import FuncAnimation
from astropy.io import fits

from src.headers.header import Header


class Array3D(np.ndarray):
    """
    Encapsulates the methods specific to three-dimensional arrays.
    """

    def __new__(cls, data):
        obj = np.asarray(data).view(cls)
        return obj

    def get_ImageHDU(self, header: Header=None):
        return fits.ImageHDU(self.data, header)

    def plot(self, fig, ax, **kwargs) -> FuncAnimation:
        """
        Plots an Array3D onto an axis.

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
            "zlim" : str, default=None. Specify the z bounds.
            "cbar_label" : str, default=None. Specify the label for the colorbar.
            "discrete_colormap" : bool, default=False. Specify if the colormap should be discrete.
            "cbar_limits" : tuple, default=None. Specify the limits of the colorbar. Essential for a discrete_colormap.
            "time_interval" : int, default=100. Specify the time interval between frames, in milliseconds.
        
        Returns
        -------
        animation : FuncAnimation
            Animation that can be saved using FuncAnimation.save.
        """
        DEFAULT_TIME_INTERVAL = 100

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

        zlim = kwargs.get("zlim", (0, self.data.shape[0]))

        if kwargs.get("cbar_limits") and not kwargs.get("discrete_colormap"):
            imshow.set_clim(*kwargs.get("cbar_limits"))
        cbar.set_label(kwargs.get("cbar_label"))
        ax.set_xlabel(kwargs.get("xlabel"))
        ax.set_ylabel(kwargs.get("ylabel"))
        if kwargs.get("xlim"):
            ax.set_xlim(*kwargs.get("xlim"))
        if kwargs.get("ylim"):
            ax.set_ylim(*kwargs.get("ylim"))
        
        ax.tick_params(axis='both', direction='in')

        def next_slice(frame_number):
            imshow.set_array(self.data[frame_number,:,:])
            cbar.update_normal(imshow)
        
        animation = FuncAnimation(fig, next_slice, frames=range(*zlim), interval=kwargs.get("time_interval", 
                                                                                            DEFAULT_TIME_INTERVAL))

        return animation
