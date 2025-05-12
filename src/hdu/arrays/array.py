import numpy as np
from astropy.io import fits
from typing import Self
from eztcolors import Colors as C

from src.headers.header import Header


class Array(np.ndarray):
    """
    Encapsulates the methods specific to arrays.
    """

    def __new__(cls, data):
        obj = np.asarray(data).view(cls)
        return obj

    def get_PrimaryHDU(self, header: Header) -> fits.PrimaryHDU:
        """
        Get the PrimaryHDU object of the Array.

        Parameters
        ----------
        header : Header
            Header of the Array.
        
        Returns
        -------
        fits.PrimaryHDU
            PrimaryHDU object of the Array.
        """
        return fits.PrimaryHDU(self.data, header)

    def get_ImageHDU(self, header: Header) -> fits.ImageHDU:
        """
        Get the ImageHDU object of the Array.

        Parameters
        ----------
        header : Header
            Header of the Array.
        
        Returns
        -------
        fits.ImageHDU
            ImageHDU object of the Array.
        """
        return fits.ImageHDU(self.data, header)

    def bin(self, bins: tuple[int, int], ignore_nans: bool=False) -> Self:
        """
        Bins an Array.

        Parameters
        ----------
        bins : tuple[int, int]
            Number of pixels to be binned together along each axis. A value of 1 results in the axis not being
            binned. The axes are in the order y, x.
        ignore_nans : bool, default=False
            Whether to ignore the nan values in the process of binning. If no nan values are present, this parameter is
            obsolete. If False, the function np.mean is used for binning whereas np.nanmean is used if True. If the nans
            are ignored, the map might increase in size as new pixels might take the place of old nans. If the nans are
            not ignored, the map might decrease in size as every new pixel that contained a nan will be made a nan also.

        Returns
        -------
        Self
            Binned Array.
        """
        assert list(bins) == list(filter(lambda val: val >= 1 and isinstance(val, int), bins)), \
            f"{C.LIGHT_RED}All values in bins must be integers greater than or equal to 1.{C.END}"
        if ignore_nans:
            func = np.nanmean
        else:
            func = np.mean

        cropped_pixels = np.array(self.shape) % np.array(bins)

        new_data = self[*[slice(None, shape - cropped_pixel) 
                          for shape, cropped_pixel in zip(self.shape, cropped_pixels)]]

        for ax, b in enumerate(bins):
            if b != 1:
                indices = list(new_data.shape)
                indices[ax:ax+1] = [new_data.shape[ax] // b, b]
                reshaped_data = new_data.reshape(indices)
                new_data = func(reshaped_data, axis=ax+1)

        return new_data
