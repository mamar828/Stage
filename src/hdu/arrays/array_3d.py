import numpy as np
from graphinglib import Heatmap
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from src.hdu.arrays.array import Array


class Array3D(Array):
    """
    Encapsulates the methods specific to three-dimensional arrays.
    """

    @property
    def plot(self) -> Heatmap:
        """
        Gives the plot of the Array3D's first slice with a Heatmap.

        Returns
        -------
        heatmap : Heatmap
            Plotted Array3D
        """
        heatmap = Heatmap(
            image=self[0,:,:],
            show_color_bar=True,
            color_map="viridis",
            origin_position="lower"
        )
        return heatmap

    def plot_mpl(self, fig: Figure, ax: Axes, **kwargs) -> FuncAnimation:
        """
        Plots an Array3D onto an axis.
        Note that the returned object needs to be assigned to a variable to stay alive.

        Parameters
        ----------
        fig : Figure
            Figure on which to plot the Array3D.
        ax : Axes
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
            Animation that can be saved using FuncAnimation.save. Assign the object to a variable to keep the animation
            running.
        """
        DEFAULT_TIME_INTERVAL = 100
        zlim = kwargs.get("zlim", (0, self.shape[0]))

        if kwargs.get("discrete_colormap"):
            viridis_cmap = plt.cm.viridis
            cbar_limits = kwargs["cbar_limits"]
            interval = (cbar_limits[1] - cbar_limits[0]) * 2
            bounds = np.linspace(*cbar_limits, interval + 1)
            cmap = ListedColormap(viridis_cmap(np.linspace(0, 1, interval)))
            norm = BoundaryNorm(bounds, cmap.N)
            imshow = ax.imshow(self[zlim[0],...], origin="lower", cmap=cmap, norm=norm)
            cbar = plt.colorbar(imshow, ticks=np.linspace(*cbar_limits, interval//2 + 1), fraction=0.046, pad=0.04)

        else:
            imshow = ax.imshow(self[zlim[0],...], origin="lower")
            cbar = plt.colorbar(imshow, fraction=0.046, pad=0.04)

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
            imshow.set_array(self[frame_number,:,:])
            cbar.update_normal(imshow)
        
        animation = FuncAnimation(fig, next_slice, frames=range(*zlim), interval=kwargs.get("time_interval", 
                                                                                            DEFAULT_TIME_INTERVAL))

        return animation

    def get_nan_cropping_slices(self) -> tuple[slice, slice, slice]:
        """
        Gives the slices for cropping the border nan values of the array.

        Returns
        -------
        slices : tuple[slice, slice]
            Slice that must be applied on the Array3D for cropping, i.e. (valid columns, valid rows).
        """
        non_nan_cols = np.any(~np.isnan(self), axis=1)
        non_nan_rows = np.any(~np.isnan(self), axis=2)

        cols = slice(np.argmax(non_nan_rows), non_nan_rows.size - np.argmax(non_nan_rows[::-1]))
        rows = slice(np.argmax(non_nan_cols), non_nan_cols.size - np.argmax(non_nan_cols[::-1]))
        return slice(None, None), cols, rows
