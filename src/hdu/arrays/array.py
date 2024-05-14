import numpy as np
from astropy.io import fits

from src.headers.header import Header


class Array(np.ndarray):
    """
    Encapsulates the methods specific to arrays.
    """

    def __new__(cls, data):
        obj = np.asarray(data).view(cls)
        return obj

    def get_ImageHDU(self, header: Header) -> fits.ImageHDU:
        """
        Get the ImageHDU object of the Array.

        Parameters
        ----------
        header : Header
            Header of the Array.
        
        Returns
        -------
        ImageHDU : fits.ImageHDU
            ImageHDU object of the Array.
        """
        return fits.ImageHDU(self.data, header)
