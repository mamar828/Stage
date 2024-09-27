from __future__ import annotations
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
        Gives the statistics of the array. Supported statistic measures are: median, mean, nbpixels stddev, skewness and
        kurtosis. If the statistics need to be computed in a region, the mask should be first applied to the Map and
        then the statistics may be computed.

        Returns
        -------
        stats : dict
            Statistic of the region. Every key is a statistic measure.
        """
        stats =  {
            "median": float(np.nanmedian(self)),
            "mean": float(np.nanmean(self)),
            "nbpixels": np.count_nonzero(~np.isnan(self)),
            "stddev": float(np.nanstd(self)),
            "skewness": scipy.stats.skew(self, axis=None, nan_policy="omit"),
            "kurtosis": scipy.stats.kurtosis(self, axis=None, nan_policy="omit")
        }
        return stats

    def get_nan_cropping_slices(self) -> tuple[slice, slice]:
        """
        Gives the slices for cropping the border nan values of the array.

        Returns
        -------
        slices : tuple[slice, slice]
            Slice that must be applied on the Array2D for cropping, i.e. (valid columns, valid rows).
        """
        non_nan_cols = np.any(~np.isnan(self), axis=0)
        non_nan_rows = np.any(~np.isnan(self), axis=1)

        cols = slice(np.argmax(non_nan_rows), non_nan_rows.size - np.argmax(non_nan_rows[::-1]))
        rows = slice(np.argmax(non_nan_cols), non_nan_cols.size - np.argmax(non_nan_cols[::-1]))
        return cols, rows
