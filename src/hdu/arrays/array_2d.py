from graphinglib import Heatmap

from src.hdu.arrays.array import Array


class Array2D(Array):
    """
    Encapsulates the methods specific to two-dimensional arrays.
    """

    @property
    def plot(self) -> Heatmap:
        """
        Gives the plot of the Array2D with a Heatmap.

        Returns
        -------
        heatmap : Heatmap
            Plotted Array2D
        """
        heatmap = Heatmap(
            image=self,
            show_color_bar=True,
            color_map="viridis",
            origin_position="lower"
        )
        return heatmap
