from numpy.typing import ArrayLike
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


class Subplot:
    """
    Object-oriented implementation of the matplotlib.axes.Axes class.
    """

    def __init__(
            self,
            font: str="Computer Modern",
            fontsize: float=12,
            tick_orientation: str="in",
            x_label: str="",
            y_label: str="",
            x_limits: tuple[float, float]=(None, None),
            y_limits: tuple[float, float]=(None, None),
            z_limits: tuple[float, float]=(None, None),
            colorbar: bool=False,
            colorbar_label: str="",
            colorbar_limits: tuple[float, float]=(None, None),
            legend: bool=False,
            legend_fontsize: float=12,
            legend_location: str="upper left",
            legend_ncols: int=1,
            frame_duration: float=0.1
    ):
        self.font = font
        self.fontsize = fontsize
        self.tick_orientation = tick_orientation
        self.x_label = x_label
        self.y_label = y_label
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.z_limits = z_limits
        self.colorbar = colorbar
        self.colorbar_label = colorbar_label
        self.colorbar_limits = colorbar_limits
        self.legend = legend
        self.legend_fontsize = legend_fontsize
        self.legend_location = legend_location
        self.legend_ncols = legend_ncols
        self.frame_duration = frame_duration

        self.plots = []
        self.imshows = []

    def plot(
            self,
            x: ArrayLike,
            y: ArrayLike,
            linecolor: str="blue",
            linestyle: str="none",
            linewidth: float=1.,
            marker: str="o",
            markersize: float=5.,
            markeredgecolor: str="none",
            markeredgewidth: float=1.,
            markerfacecolor: str="blue",
            label: str="",
            alpha: float=1.,
            fillstyle: str="full"
    ):
        self.plots.append({
            "x" : x,
            "y" : y,
            "linecolor" : linecolor,
            "linestyle" : linestyle,
            "linewidth" : linewidth,
            "marker" : marker,
            "markersize" : markersize,
            "markeredgecolor" : markeredgecolor,
            "markeredgewidth" : markeredgewidth,
            "markerfacecolor" : markerfacecolor,
            "label" : label,
            "alpha" : alpha,
            "fillstyle" : fillstyle
        })
    
    def imshow(
            self,
            X: ArrayLike,
            cmap: str="viridis",
            vmin: float=None,
            vmax: float=None,
            origin: str="upper",
            alpha: float=1.
    ):
        self.imshows.append({
            "X" : X,
            "cmap" : cmap,
            "vmin" : vmin,
            "vmax" : vmax,
            "origin" : origin,
            "alpha" : alpha,
        })

    def show(self, ax: Axes):
        plt.rcParams["font.serif"] = self.font
        plt.rcParams["font.size"] = self.fontsize
        for plot in self.plots:
            ax.plot(
                plot["x"],
                plot["y"],
                color=plot["linecolor"],
                linestyle=plot["linestyle"],
                linewidth=plot["linewidth"],
                marker=plot["marker"],
                markersize=plot["markersize"],
                markeredgecolor=plot["markeredgecolor"],
                markeredgewidth=plot["markeredgewidth"],
                markerfacecolor=plot["markerfacecolor"],
                label=plot["label"],
                alpha=plot["alpha"],
                fillstyle=plot["fillstyle"]
            )
        
        for imshow in self.imshows:
            ax.imshow(
                imshow["X"],
                cmap=imshow["cmap"],
                vmin=imshow["vmin"],
                vmax=imshow["vmax"],
                origin=imshow["origin"],
                alpha=imshow["alpha"]
            )

        ax.tick_params(axis="both", direction=self.tick_orientation, labelsize=self.fontsize)
        ax.set_xlabel = self.x_label
        ax.set_ylabel = self.y_label
        ax.set_xlim(*self.x_limits)
        ax.set_ylim(*self.y_limits)
        # self.colorbar = colorbar
        # self.colorbar_label = colorbar_label
        # self.colorbar_limits = colorbar_limits
        if self.legend:
            ax.legend(
                loc=self.legend_location,
                fontsize=self.legend_fontsize,
                ncols=self.legend_ncols
            )

        # self.z_limits
        # self.frame_duration = frame_duration
