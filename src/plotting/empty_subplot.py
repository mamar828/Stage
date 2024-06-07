import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class EmptySubplot:
    def __init__(self):
        self.font = "Times New Roman"
        self.fontsize = 12

    def show(self, ax: Axes):
        plt.rcParams["font.sans-serif"] = self.font
        plt.rcParams["font.size"] = self.fontsize
        ax.text(
            x=0.3,
            y=0.5,
            s="Empty subplot"
        )
