from __future__ import annotations
import numpy as np
from astropy.io import fits
from typing import Self, Any
from eztcolors import Colors as C

from src.hdu.fits_file import FitsFile
from src.hdu.arrays.array_2d import Array2D
from src.hdu.arrays.array_3d import Array3D
from src.hdu.maps.map import Map, MapCO
from src.spectrums.spectrum import Spectrum
from src.spectrums.spectrum_co import SpectrumCO
from src.headers.header import Header
from src.base_objects.silent_none import SilentNone


class Cube(FitsFile):
    """
    Encapsulates the methods specific to data cubes.
    """
    spectrum_type, map_type = Spectrum, Map

    def __init__(self, data: Array3D, header: Header = SilentNone()):
        """
        Initialize a Cube object.

        Parameters
        ----------
        data : Array3D
            The values of the Cube.
        header : Header, default=SilentNone()
            The header of the Cube.
        """
        self.data = Array3D(data)
        self.header = header

    def __eq__(self, other: Any) -> bool:
        same_array = np.allclose(self.data, other.data, equal_nan=True)
        same_header = self.header == other.header
        return same_array and same_header

    def __getitem__(self, slices: tuple[slice | int]) -> Spectrum | SpectrumCO | Map | MapCO | Self:
        if not all([isinstance(s, (int, slice)) for s in slices]):
            raise TypeError(f"{C.LIGHT_RED}Every slice element must be an int or a slice.{C.END}")
        int_slices = [isinstance(slice_, int) for slice_ in slices]
        if int_slices.count(True) == 1:
            map_header = self.header.flatten(axis=int_slices.index(True))
            return self.map_type(data=Array2D(self.data[slices]), header=map_header)
        elif int_slices.count(True) == 2:
            first_int_i = int_slices.index(True)
            map_header = self.header.flatten(axis=first_int_i)
            spectrum_header = map_header.flatten(axis=(int_slices.index(True, first_int_i+1)))
            return self.spectrum_type(data=self.data[slices], header=spectrum_header)
        elif int_slices.count(True) == 3:
            return self.data[slices]
        else:
            return self.__class__(self.data[slices], self.header.crop_axes(slices))
    
    def __iter__(self) -> Self:
        self.iter_n = -1
        return self
    
    def __next__(self) -> Self:
        self.iter_n += 1
        if self.iter_n >= self.data.shape[1]:
            raise StopIteration
        else:
            return self[:,self.iter_n,:]

    def copy(self) -> Self:
        return self.__class__(self.data.copy(), self.header.copy())
    
    @classmethod
    def load(cls, filename: str) -> Cube:
        """
        Loads a Cube from a .fits file.

        Parameters
        ----------
        filename : str
            Name of the file to load.
        
        Returns
        -------
        Cube
            An instance of the given class containing the file's contents.
        """
        fits_object = fits.open(filename)[0]
        cube = cls(
            Array3D(fits_object.data),
            Header(fits_object.header)
        )
        return cube

    def save(self, filename: str, overwrite: bool = False):
        """
        Saves a Cube to a file.

        Parameters
        ----------
        filename : str
            Filename in which to save the Cube.
        overwrite : bool, default=False
            Whether the file should be forcefully overwritten if it already exists.
        """
        super().save(filename, fits.HDUList([self.data.get_PrimaryHDU(self.header)]), overwrite)

    def bin(self, bins: tuple[int, int, int], ignore_nans: bool = False) -> Self:
        """
        Bins a Cube.

        Parameters
        ----------
        bins : tuple[int, int, int]
            Number of pixels to be binned together along each axis. A value of 1 results in the axis not being binned.
            The axes are in the order z, y, x.
        ignore_nans : bool, default=False
            Whether to ignore the nan values in the process of binning. If no nan values are present, this parameter is
            obsolete. If False, the function np.mean is used for binning whereas np.nanmean is used if True. If the nans
            are ignored, the cube might increase in size as pixels will take the place of nans. If the nans are not
            ignored, the cube might decrease in size as every new pixel that contained a nan will be made a nan also.

        Returns
        -------
        Self
            Binned Cube.
        """
        return self.__class__(self.data.bin(bins, ignore_nans), self.header.bin(bins, ignore_nans))

    def invert_axis(self, axis: int) -> Self:
        """
        Inverts the elements' order along an axis.

        Parameters
        ----------
        axis : int
            Axis whose order must be flipped. 0, 1, 2 correspond to z, y, x respectively.

        Returns
        -------
        Self
            Cube with the newly axis-flipped Data_cube.
        """
        return self.__class__(np.flip(self.data, axis=axis), self.header.invert_axis(axis))

    def swap_axes(self, axis_1: int, axis_2: int) -> Self:
        """
        Swaps a Cube's axes.
        
        Parameters
        ----------
        axis_1: int
            Source axis.
        axis_2: int
            Destination axis.
        
        Returns
        -------
        Self
            Cube with the switched axes.
        """
        new_data = self.data.swapaxes(axis_1, axis_2)
        new_header = self.header.swap_axes(axis_1, axis_2)
        return self.__class__(new_data, new_header)

    def crop_nans(self) -> Self:
        """
        Crops the nan values at the borders of the Cube.

        Returns
        -------
        Self
            Cube with the nan values removed.
        """
        return self[self.data.get_nan_cropping_slices()]
