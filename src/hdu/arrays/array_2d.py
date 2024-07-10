from graphinglib import Heatmap
import numpy as np
import scipy

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

    def get_statistics(self) -> dict:
        """
        Get the statistics of the array. Supported statistic measures are: median, mean, nbpixels stddev, skewness and
        kurtosis. If the statistics need to be computed in a region, the mask should be first applied to the Map and
        then the statistics may be computed.

        Returns
        -------
        dict: statistics of the region. Every key is a statistic measure.
        """
        stats =  {
            "median": np.nanmedian(self),
            "mean": np.nanmean(self),
            "nbpixels": np.count_nonzero(~np.isnan(self)),
            "stddev": np.nanstd(self),
            "skewness": scipy.stats.skew(self, axis=None, nan_policy="omit"),
            "kurtosis": scipy.stats.kurtosis(self, axis=None, nan_policy="omit")
        }
        return stats
