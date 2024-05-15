from __future__ import annotations
import numpy as np
import pyregion
from astropy.io import fits
from eztcolors import Colors as C

from src.hdu.fits_file import FitsFile
from src.hdu.arrays.array_2d import Array2D
from src.headers.header import Header


class Map(FitsFile):
    """
    Encapsulates the necessary methods to compare and treat maps.
    """

    def __init__(self, data: Array2D, uncertainties: Array2D=np.NAN, header: Header=None):
        """
        Initialize a Map object.

        Parameters
        ----------
        data : Array2D
            The values of the Map.
        uncertainties : Array2D, default=np.NAN
            The uncertainties of the Map.
        header : Header, default=None
            Header of the Map.
        """
        self.data = data
        self.uncertainties = uncertainties
        self.header = header

    def __add__(self, other):
        if isinstance(other, Map):
            self.assert_shapes(other)
            return Map(
                Array2D(self.data + other.data),
                Array2D(self.uncertainties + other.uncertainties),
                self.header
            )
        elif isinstance(other, (int, float)):
            return Map(
                Array2D(self.data + other),
                np.NAN,
                self.header
            )
        else:
            raise TypeError(
                f"{C.LIGHT_RED}unsupported operand type(s) for +: 'Map' and '{type(other).__name__}'{C.END}")
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Map):
            self.assert_shapes(other)
            return Map(
                Array2D(self.data - other.data),
                Array2D(self.uncertainties + other.uncertainties),
                self.header
            )
        elif isinstance(other, (int, float)):
            return Map(
                Array2D(self.data - other),
                np.NAN,
                self.header
            )
        else:
            raise TypeError(
                f"{C.LIGHT_RED}unsupported operand type(s) for -: 'Map' and '{type(other).__name__}'{C.END}")
    
    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Map):
            self.assert_shapes(other)
            return Map(
                Array2D(self.data * other.data),
                Array2D(((self.uncertainties / self.data) + (other.uncertainties / other.data)) 
                * self.data * other.data),
                self.header
            )
        elif isinstance(other, (int, float)):
            return Map(
                Array2D(self.data * other),
                np.NAN,
                self.header
            )
        else:
            raise TypeError(
                f"{C.LIGHT_RED}unsupported operand type(s) for *: 'Map' and '{type(other).__name__}'{C.END}")

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, Map):
            self.assert_shapes(other)
            return Map(
                Array2D(self.data / other.data),
                Array2D(((self.uncertainties / self.data) + (other.uncertainties / other.data)) \
                * self.data / other.data)
            )
        elif isinstance(other, (int, float)):
            return Map(
                Array2D(self.data / other),
                np.NAN
            )
        else:
            raise TypeError(
                f"{C.LIGHT_RED}unsupported operand type(s) for /: 'Map' and '{type(other).__name__}'{C.END}")
    
    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __getitem__(self, key: slice) -> Map:
        return Map(
            self.data[key],
            self.uncertainties[key]
        )
    
    def __str__(self):
        return (f"Value : {True if isinstance(self.data, Array2D) else False}, "
              + f"Uncertainty : {True if isinstance(self.uncertainties, Array2D) else False}")
    
    @property
    def shape(self):
        return self.data.shape

    @classmethod
    def load(cls, filename) -> Map:
        """
        Loads a Map from a file.

        Parameters
        ----------
        filename : str
            Filename from which to load the Map.
        
        Returns
        -------
        map : Map
            Loaded Map.
        """
        hdu_list = fits.open(filename)
        data = Array2D(hdu_list[0].data)
        uncertainties = np.NAN
        if len(hdu_list) == 2:
            uncertainties = Array2D(hdu_list[1].data)
        return cls(data, uncertainties, Header(hdu_list[0].header))
        # return cls(data, uncertainties, Header.from_header(hdu_list[0].header))
    
    def save(self, filename, overwrite=False):
        """
        Saves a Map to a file.

        Parameters
        ----------
        filename : str
            Filename in which to save the Map.
        overwrite : bool, default=False
            Whether the file should be forcefully overwritten if it already exists.
        """
        hdu_list = fits.HDUList([])
        hdu_list.append(self.data.get_PrimaryHDU(self.header))
        if self.uncertainties is not None:
            hdu_list.append(self.uncertainties.get_ImageHDU(self.header))
        super().save(filename, hdu_list, overwrite)

    def assert_shapes(self, other: Map):
        """
        Asserts that two Maps have the same shape.

        Parameters
        ----------
        other : Map
            Map to compare the current map with.
        """
        assert self.shape == other.shape, \
            f"{C.LIGHT_RED}Both Maps should have the same shapes. Current are {self.shape} and {other.shape}.{C.END}"
        
    def get_masked_region(self, region: pyregion.core.ShapeList) -> Map:
        """
        Gets the Map withing a region.

        Parameters
        ----------
        region : pyregion.core.ShapeList
            Region that will be kept in the final Map.
        
        Returns
        -------
        map : Map
            Masked Map.
        """
        mask = region.get_mask(fits.PrimaryHDU(self.data))
        mask = np.where(mask == False, np.nan, 1)
        return Map(
            self.data * mask,
            self.uncertainties * mask
        )
