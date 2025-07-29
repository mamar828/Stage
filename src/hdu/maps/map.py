from __future__ import annotations
import numpy as np
import pyregion
from astropy.io import fits
import scipy
from uncertainties import ufloat
from reproject import reproject_interp
from typing import Self
from colorist import BrightColor as C
from logging import warning

from src.hdu.fits_file import FitsFile
from src.hdu.arrays.array_2d import Array2D
from src.headers.header import Header
from src.spectrums.spectrum import Spectrum
from src.spectrums.spectrum_co import SpectrumCO
from src.base_objects.mathematical_object import MathematicalObject
from src.base_objects.silent_none import SilentNone


class Map(FitsFile, MathematicalObject):
    """
    Encapsulates the necessary methods to compare and treat maps.

    Uncertainty propagation is done automatically using linear error propagation theory (similar to what is done by the
    uncertainties package). The formulaes are given in
    Bevington, P. R., & Robinson, D. K. (2003). Data reduction and error analysis. McGrawâ€“Hill, New York.
    Note: the formula for natural logarithm propagation is incorrect, no b factor should be present.
    """
    spectrum_type = Spectrum

    def __init__(self, data: Array2D, uncertainties: Array2D = SilentNone(), header: Header = SilentNone()):
        """
        Initialize a Map object.

        Parameters
        ----------
        data : Array2D
            The values of the Map.
        uncertainties : Array2D, default=SilentNone()
            The uncertainties of the Map.
        header : Header, default=SilentNone()
            Header of the Map.
        """
        if isinstance(uncertainties, Header):
            raise TypeError(
                f"{C.RED}Uncertainties must be an Array2D object, not a Header.{C.OFF}"
            )

        self.data = Array2D(data)
        self.uncertainties = Array2D(uncertainties) if not type(uncertainties) == SilentNone else uncertainties
        self.header = header

    def __add__(self, other: Map | int | float | np.ndarray) -> Self:
        if isinstance(other, Map):
            self.assert_shapes(other)
            return self.__class__(
                self.data + other.data,
                np.sqrt(self.uncertainties**2 + other.uncertainties**2),
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
                f"{C.RED}unsupported operand type(s) for +: 'Map' and '{type(other).__name__}'{C.OFF}")

    def __sub__(self, other: Map | int | float | np.ndarray) -> Self:
        if isinstance(other, Map):
            self.assert_shapes(other)
            return self.__class__(
                self.data - other.data,
                np.sqrt(self.uncertainties**2 + other.uncertainties**2),
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
                f"{C.RED}unsupported operand type(s) for -: 'Map' and '{type(other).__name__}'{C.OFF}")

    def __mul__(self, other: Map | int | float | np.ndarray) -> Self:
        if isinstance(other, Map):
            self.assert_shapes(other)
            return self.__class__(
                self.data * other.data,
                self.data*other.data * np.sqrt((self.uncertainties/self.data)**2 + (other.uncertainties/other.data)**2),
                self.header
            )
        elif isinstance(other, (int, float)) or (isinstance(other, np.ndarray) and other.size == 1):
            return self.__class__(
                self.data * other,
                self.uncertainties * abs(other),
                self.header
            )
        else:
            raise NotImplementedError(
                f"{C.RED}unsupported operand type(s) for *: 'Map' and '{type(other).__name__}'{C.OFF}")

    def __truediv__(self, other: Map | int | float | np.ndarray) -> Self:
        if isinstance(other, Map):
            self.assert_shapes(other)
            return self.__class__(
                self.data / other.data,
                self.data/other.data * np.sqrt((self.uncertainties/self.data)**2 + (other.uncertainties/other.data)**2),
                self.header
            )
        elif isinstance(other, (int, float)) or (isinstance(other, np.ndarray) and other.size == 1):
            return self.__class__(
                self.data / other,
                self.uncertainties / abs(other),
                self.header
            )
        else:
            raise NotImplementedError(
                f"{C.RED}unsupported operand type(s) for /: 'Map' and '{type(other).__name__}'{C.OFF}")

    def __pow__(self, power: int | float | np.ndarray) -> Self:
        if isinstance(power, (int, float)) or (isinstance(power, np.ndarray) and power.size == 1):
            # cast to float type to solve the integers to negative integer powers ValueError
            pow_data = self.data.astype(float)**power
            return self.__class__(
                pow_data,
                pow_data * np.abs(power * self.uncertainties / self.data),
                self.header
            )
        else:
            raise NotImplementedError(
                f"{C.RED}unsupported operand type(s) for **: 'Map' and '{type(power).__name__}'{C.OFF}")

    def __abs__(self) -> Self:
        return self.__class__(
            np.abs(self.data),
            self.uncertainties,
            self.header
        )

    def __getitem__(self, slices: tuple[slice | int]) -> Array2D | Spectrum | Map:
        int_slices = [isinstance(slice_, int) for slice_ in slices]
        if int_slices.count(True) == 1:
            warning(f"{C.YELLOW}Slicing a Map with a single integer will result in a spectrum with a SilentNone header."
                    + C.OFF)
            return self.spectrum_type(data=self.data[slices], header=SilentNone())
        elif int_slices.count(True) == 2:
            return self.data[slices]
        else:
            return self.__class__(
                self.data[slices],
                self.uncertainties[slices],
                header=self.header.slice(slices)
            )

    def __iter__(self) -> Self:
        self.iter_n = -1
        return self

    def __next__(self) -> Self:
        self.iter_n += 1
        if self.iter_n >= self.data.shape[1]:
            raise StopIteration
        else:
            return self[:,self.iter_n]

    def __str__(self) -> str:
        return (f"Value : {True if isinstance(self.data, Array2D) else False}, "
              + f"Uncertainty : {self.has_uncertainties}")

    @property
    def shape(self) -> np.ndarray:
        return self.data.shape

    @property
    def has_uncertainties(self) -> bool:
        return not isinstance(self.uncertainties, SilentNone)

    @classmethod
    def load(cls, filename: str) -> Map:
        """
        Loads a Map from a file.

        Parameters
        ----------
        filename : str
            Filename from which to load the Map.

        Returns
        -------
        Map
            An instance of the given class containing the file's contents.
        """
        hdu_list = fits.open(filename)
        data = Array2D(hdu_list[0].data)
        uncertainties = SilentNone()
        if len(hdu_list) > 1:
            uncertainties = Array2D(hdu_list[1].data)
        if len(hdu_list) > 2:
            print(f"{C.YELLOW}Warning: the given file {filename} contains more than two HDU elements. Only the first"
                 +f" two will be opened.{C.OFF}")
        if len(data.shape) != 2:
            raise TypeError("The provided data is not two-dimensional.")
        return cls(data, uncertainties, Header(hdu_list[0].header))

    @property
    def hdu_list(self) -> fits.HDUList:
        """
        Gives the Map's HDUList.

        Returns
        -------
        fits.HDUList
            List of PrimaryHDU and ImageHDU objects representing the Map.
        """
        hdu_list = fits.HDUList([])
        hdu_list.append(self.data.get_PrimaryHDU(self.header))
        if self.has_uncertainties:
            hdu_list.append(self.uncertainties.get_ImageHDU(self.header))
        return hdu_list

    def save(self, filename: str, overwrite: bool = False):
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
            f"{C.RED}Both Maps should have the same shapes. Current are {self.shape} and {other.shape}.{C.OFF}"

    def bin(self, bins: tuple[int, int], ignore_nans: bool = False) -> Self:
        """
        Bins a Map.

        Parameters
        ----------
        bins : tuple[int, int]
            Number of pixels to be binned together along each axis. A value of 1 results in the axis not being
            binned. The axes are in the order y, x.
        ignore_nans : bool, default=False
            Whether to ignore the nan values in the process of binning. If no nan values are present, this parameter is
            obsolete. If False, the function np.mean is used for binning whereas np.nanmean is used if True. If the nans
            are ignored, the map might increase in size as pixels will take the place of nans. If the nans are not
            ignored, the map might decrease in size as every new pixel that contained a nan will be made a nan also.

        Returns
        -------
        Self
            Binned Map.
        """
        return self.__class__(
            self.data.bin(bins, ignore_nans),
            self.uncertainties.bin(bins, ignore_nans),
            self.header.bin(bins)
        )

    def crop_nans(self) -> Self:
        """
        Crops the nan values at the borders of the Map.

        Returns
        -------
        Self
            Map with the nan values removed.
        """
        return self[self.data.get_nan_cropping_slices()]

    def log(self) -> Self:
        """
        Computes the natural logarithm of the Map.

        Returns
        -------
        Self
            ln(self), with uncertainties.
        """
        return self.__class__(
            np.log(self.data),
            self.uncertainties / self.data,
            self.header
        )

    def exp(self) -> Self:
        """
        Computes the exponent of the Map.

        Returns
        -------
        Self
            e**(self), with uncertainties.
        """
        exp_data = np.exp(self.data)
        return self.__class__(
            exp_data,
            exp_data * self.uncertainties,
            self.header
        )

    def num_to_nan(self, num: float = 0) -> Self:
        """
        Converts a number to np.NAN and changes the uncertainties accordingly.

        Parameters
        ----------
        num : float, default=0
            Value to convert to np.NAN.

        Returns
        -------
        Self
            Map with converted values to NANs. Note that the mask created for the data is used also for the
            uncertainties (indices where num was encountered in the data array will also be replaced with NAN in the
            uncertainties array).
        """
        mask = self.data == num
        map_ = self.copy()
        map_.data[mask] = np.NAN
        map_.uncertainties[mask] = np.NAN
        return map_

    @FitsFile.silence_function
    def get_masked_region(self, region: pyregion.Shape | pyregion.ShapeList) -> Self:
        """
        Gives the Map within a region.

        Parameters
        ----------
        region : pyregion.Shape | pyregion.ShapeList
            Region that will be kept in the final Map. If None, the whole map is returned.

        Returns
        -------
        Self
            Masked Map.
        """
        if region:
            if isinstance(region, pyregion.Shape):
                region = pyregion.ShapeList([region])
            if self.header:
                mask = region.get_mask(self.data.get_PrimaryHDU(self.header))
            else:
                mask = region.get_mask(shape=self.data.shape)
            mask = np.where(mask == False, np.nan, 1)
        else:
            mask = np.ones_like(self.data)
        return self.__class__(
            self.data * mask,
            self.uncertainties * mask,
            self.header
        )

    def get_statistics(self, region: pyregion.core.ShapeList=None) -> dict:
        """
        Gives the statistics of the map's data. Supported statistic measures are: median, mean, nbpixels stddev,
        skewness and kurtosis. The statistics may be computed in a region, if one is given. This method is for
        convenience and uses the lower-level Array2D.get_statistics method.

        Arguments
        ---------
        region: pyregion.core.ShapeList, default=None
            If present, region in which the statistics need to be calculated.

        Returns
        -------
        dict
            Statistic of the region. Every key is a statistic measure.
        """
        reg_map = self.get_masked_region(region)

        uncertainties_array = np.vectorize(lambda data, unc: ufloat(data, unc))(reg_map.data, reg_map.uncertainties)

        stats =  {
            "median": np.nanmedian(uncertainties_array)[()],
            "mean": np.nanmean(uncertainties_array)[()],
            "nbpixels": np.count_nonzero(~np.isnan(reg_map.data)),
            "stddev": float(np.nanstd(reg_map.data)),
            "skewness": scipy.stats.skew(reg_map.data, axis=None, nan_policy="omit"),
            "kurtosis": scipy.stats.kurtosis(reg_map.data, axis=None, nan_policy="omit")
        }

        return stats

    @FitsFile.silence_function
    def get_reprojection_on(self, header: Header) -> Self:
        """
        Gives the reprojection of the Map on another Map's coordinate system. This coordinate matching allows for
        operations between differently sized/aligned Maps.

        Parameters
        ----------
        header : Header
            Reference header to which the Map will be reprojected.

        Returns
        -------
        Self
            Newly aligned Map.
        """
        if not isinstance(header, fits.Header):
            raise TypeError(f"{C.RED}Header must be an astropy.io.fits.Header object.{C.OFF}")
        data_reprojection = Array2D(reproject_interp(
            input_data=self.data.get_PrimaryHDU(self.header),
            output_projection=header,
            return_footprint=False,
            order="nearest-neighbor"
        ))
        if self.has_uncertainties:
            uncertainties_reprojection = Array2D(reproject_interp(
                input_data=self.uncertainties.get_PrimaryHDU(self.header),
                output_projection=header,
                return_footprint=False,
                order="nearest-neighbor"
            ))
        else:
            uncertainties_reprojection = self.uncertainties
        return self.__class__(
            data_reprojection,
            uncertainties_reprojection,
            header=header.copy()
        )


class MapCO(Map):
    """
    This class allows to output SpectrumCO when slicing a Map previously constructed with a CubeCO.
    """
    spectrum_type = SpectrumCO
