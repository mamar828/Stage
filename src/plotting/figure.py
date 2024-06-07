import numpy as np
import matplotlib.pyplot as plt

from src.plotting.subplot import Subplot
from src.plotting.empty_subplot import EmptySubplot


class Figure:
    """
    Object-oriented implementation of the matplotlib.pyplot.figure class.
    """

    def __init__(
            self,
            subplots: tuple[1, 1],
            figsize: tuple[float, float]=(8, 6),
            font: str="Computer Modern",
            fontsize: float=12,
            x_label: str="",
            y_label: str="",
            share_x: bool=False,
            share_y: bool=False,
            tight_layout: bool=False
    ):
        self.subplots = np.full(shape=subplots, fill_value=EmptySubplot())
        self.figsize = figsize
        self.font = font
        self.fontsize = fontsize
        self.x_label = x_label
        self.y_label = y_label
        self.share_x = share_x
        self.share_y = share_y
        self.tight_layout = tight_layout

    def __getitem__(self, key: tuple[int | int]) -> Subplot:
        return self.subplots[key]
    
    def __setitem__(self, key: tuple[int, int], value: Subplot):
        self.subplots[key] = value
    
    def show(self):
        # Clear previously existing figures and axes
        plt.clf()
        plt.cla()

        fig, axs = plt.subplots(*self.subplots.shape, sharex=self.share_x, sharey=self.share_y, figsize=self.figsize)
        if self.tight_layout:
            plt.tight_layout()

        for i, row in enumerate(self.subplots):
            for j, subplot in enumerate(row):
                subplot.show(axs[i, j])

        plt.rcParams["font.serif"] = self.font
        plt.rcParams["font.size"] = self.fontsize
        fig.supxlabel(self.x_label)
        fig.supylabel(self.y_label)

        plt.show()

    def save(self, filename: str, dpi: int=600, bbox_inches: str="tight"): ...


x_space = np.linspace(0, 10, 100)
subplot_01 = Subplot(
    x_label="x",
    y_label="y"
)
subplot_01.plot(
    x_space,
    np.sin(x_space),
    markerfacecolor="red",
    linestyle="-",
    markersize=2
)

fig = Figure(
    subplots=(3,2),
    y_label="alloha !"
)
fig[0,1] = subplot_01
fig[2,1] = subplot_01
fig.show()
