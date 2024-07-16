from __future__ import annotations
import numpy as np
import pyregion
from astropy.io import fits
from eztcolors import Colors as C
import scipy
from uncertainties import ufloat
from reproject import reproject_interp
from typing import Self

from src.hdu.fits_file import FitsFile
from src.hdu.arrays.array_2d import Array2D
from src.headers.header import Header
from src.spectrums.spectrum import Spectrum
from src.spectrums.spectrum_co import SpectrumCO
from src.base_objects.mathematical_object import MathematicalObject


class Map(FitsFile, MathematicalObject):
    """
    Encapsulates the necessary methods to compare and treat maps.
    """
    spectrum_type = Spectrum

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
            return self.__class__(
                self.data + other.data,
                self.uncertainties + other.uncertainties,
                self.header
            )
        elif isinstance(other, (int, float)) or (isinstance(other, np.ndarray) and other.size == 1):
            return self.__class__(
                self.data + other,
                self.uncertainties,
                self.header
            )
        else:
            raise NotImplementedError(
                f"{C.LIGHT_RED}unsupported operand type(s) for +: 'Map' and '{type(other).__name__}'{C.END}")
    
    def __sub__(self, other):
        if isinstance(other, Map):
            self.assert_shapes(other)
            return self.__class__(
                self.data - other.data,
                self.uncertainties + other.uncertainties,
                self.header
            )
        elif isinstance(other, (int, float)) or (isinstance(other, np.ndarray) and other.size == 1):
            return self.__class__(
                self.data - other,
                self.uncertainties,
                self.header
            )
        else:
            raise NotImplementedError(
                f"{C.LIGHT_RED}unsupported operand type(s) for -: 'Map' and '{type(other).__name__}'{C.END}")
    
    def __mul__(self, other):
        if isinstance(other, Map):
            self.assert_shapes(other)
            return self.__class__(
                self.data * other.data,
                ((self.uncertainties / self.data) + (other.uncertainties / other.data)) * self.data * other.data,
                self.header
            )
        elif isinstance(other, (int, float)) or (isinstance(other, np.ndarray) and other.size == 1):
            return self.__class__(
                self.data * other,
                self.uncertainties * other,
                self.header
            )
        else:
            raise NotImplementedError(
                f"{C.LIGHT_RED}unsupported operand type(s) for *: 'Map' and '{type(other).__name__}'{C.END}")

    def __truediv__(self, other):
        if isinstance(other, Map):
            self.assert_shapes(other)
            return self.__class__(
                self.data / other.data,
                ((self.uncertainties / self.data) + (other.uncertainties / other.data)) * self.data / other.data,
                self.header
            )
        elif isinstance(other, (int, float)) or (isinstance(other, np.ndarray) and other.size == 1):
            return self.__class__(
                self.data / other,
                self.uncertainties / other,
                self.header
            )
        else:
            raise NotImplementedError(
                f"{C.LIGHT_RED}unsupported operand type(s) for /: 'Map' and '{type(other).__name__}'{C.END}")
    
    def __pow__(self, power):
        if isinstance(power, (int, float)) or (isinstance(power, np.ndarray) and power.size == 1):
            return self.__class__(
                self.data ** power,
                np.abs(self.uncertainties / self.data * power * self.data**power),
                self.header
            )
        else:
            raise NotImplementedError(
                f"{C.LIGHT_RED}unsupported operand type(s) for **: 'Map' and '{type(power).__name__}'{C.END}")

    def __getitem__(self, slices: tuple[slice | int]) -> Spectrum | SpectrumCO | Map:
        int_slices = [isinstance(slice_, int) for slice_ in slices]
        if int_slices.count(True) == 1:
            spectrum_header = self.header.flatten(axis=int_slices.index(True))
            return self.spectrum_type(data=self.data[slices], header=spectrum_header)
        elif int_slices.count(True) == 2:
            return self.data[slices]
        else:
            return self.__class__(
                self.data[slices],
                self.uncertainties[slices] if isinstance(self.uncertainties, Array2D) else np.NAN,
                header=self.header.crop_axes(slices)
            )

    def __iter__(self):
        self.iter_n = -1
        return self
    
    def __next__(self):
        self.iter_n += 1
        if self.iter_n >= self.data.shape[1]:
            raise StopIteration
        else:
            return self[:,self.iter_n]

    def __str__(self):
        return (f"Value : {True if isinstance(self.data, Array2D) else False}, "
              + f"Uncertainty : {True if isinstance(self.uncertainties, Array2D) else False}")

    @property
    def shape(self):
        return self.data.shape

    @classmethod
    def load(cls, filename: str) -> Self:
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
    
    @property
    def hdu_list(self) -> fits.HDUList:
        """
        Gives the Map's HDUList.
        
        Returns
        -------
        hdu_list : fits.HDUList
            List of PrimaryHDU and ImageHDU objects representing the Map.
        """
        hdu_list = fits.HDUList([])
        hdu_list.append(self.data.get_PrimaryHDU(self.header))
        if self.uncertainties is not np.NAN:
            hdu_list.append(self.uncertainties.get_ImageHDU(self.header))
        return hdu_list

    def save(self, filename: str, overwrite: bool=False):
        """
        Saves a Map to a file.

        Parameters
        ----------
        filename : str
            Filename in which to save the Map.
        overwrite : bool, default=False
            Whether the file should be forcefully overwritten if it already exists.
        """
        super().save(filename, self.hdu_list, overwrite)

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
        
    @FitsFile.silence_function
    def get_masked_region(self, region: pyregion.core.ShapeList) -> Self:
        """
        Gets the Map within a region.

        Parameters
        ----------
        region : pyregion.core.ShapeList
            Region that will be kept in the final Map. If None, the whole map is returned.
        
        Returns
        -------
        map : Map
            Masked Map.
        """
        if region:
            if self.header:
                mask = region.get_mask(fits.PrimaryHDU(self.data, self.header))
            else:
                mask = region.get_mask(shape=self.data.shape)
            mask = np.where(mask == False, np.nan, 1)
        else:
            mask = np.ones_like(self.data)
        return self.__class__(
            self.data * mask,
            self.uncertainties * mask
        )

    def get_statistics(self, region: pyregion.core.ShapeList=None) -> dict:
        """
        Get the statistics of the map's data. Supported statistic measures are: median, mean, nbpixels stddev, skewness
        and kurtosis. The statistics may be computed in a region, if one is given. This method is for convenience and
        uses the lower-level Array2D.get_statistics method.

        Arguments
        ---------
        region: pyregion.core.ShapeList, default=None. If present, region in which the statistics need to be calculated.

        Returns
        -------
        dict: statistics of the region. Every key is a statistic measure.
        """
        reg_map = self.get_masked_region(region)
        
        uncertainties_array = np.vectorize(lambda data, unc: ufloat(data, unc))(reg_map.data, reg_map.uncertainties)

        stats =  {
            "median": np.nanmedian(uncertainties_array),
            "mean": np.nanmean(uncertainties_array),
            "nbpixels": np.count_nonzero(~np.isnan(reg_map.data)),
            "stddev": np.nanstd(reg_map.data),
            "skewness": scipy.stats.skew(reg_map.data, axis=None, nan_policy="omit"),
            "kurtosis": scipy.stats.kurtosis(reg_map.data, axis=None, nan_policy="omit")
        }

        return stats

    @FitsFile.silence_function
    def get_reprojection_on(self, other: Map) -> Self:
        """
        Gives the reprojection of the Map on another Map's coordinate system. This coordinate matching allows for
        operations between differently sized/aligned Maps.
        
        Parameters
        ----------
        other : Map
            Reference Map to project on.
            
        Returns
        -------
        reprojected map : Map
            Newly aligned Map.
        """
        data_reprojection = Array2D(reproject_interp(
            input_data=self.data.get_PrimaryHDU(self.header),
            output_projection=other.header,
            return_footprint=False,
            order="nearest-neighbor"
        ))
        if self.uncertainties is not np.NAN:
            uncertainties_reprojection = Array2D(reproject_interp(
                input_data=self.uncertainties.get_PrimaryHDU(self.header),
                output_projection=other.header,
                return_footprint=False,
                order="nearest-neighbor"
            ))
        else:
            uncertainties_reprojection = self.uncertainties
        return self.__class__(
            data_reprojection,
            uncertainties_reprojection,
            header=other.header
        )


class MapCO(Map):
    """
    This class allows to output SpectrumCO when slicing a Map previously constructed with a CubeCO.
    """
    spectrum_type = SpectrumCO
