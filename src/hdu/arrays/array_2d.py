from graphinglib import Heatmap
import numpy as np
import scipy
from uncertainties import ufloat
import pyregion
from typing_extensions import Self
from astropy.io import fits

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

    def get_masked_region(self, region: pyregion.core.ShapeList) -> Self:
        raise NotImplementedError
        """
        Gets a masked array.

        Parameters
        ----------
        region : pyregion.core.ShapeList
            Region to keep in the array. If None, the whole array is returned.
        
        Returns
        -------
        map : Map
            Masked Array2D.
        """
        if region:
            mask = region.get_mask(fits.PrimaryHDU(self.data))
            mask = np.where(mask == False, np.NAN, 1)
        else:
            mask = np.ones_like(self.data)
        return self * mask

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


