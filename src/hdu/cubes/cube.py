from __future__ import annotations
import numpy as np
from astropy.io import fits
from eztcolors import Colors as C

from src.hdu.fits_file import FitsFile
from src.hdu.arrays.array_2d import Array2D
from src.hdu.arrays.array_3d import Array3D
from src.hdu.maps.map import Map
from src.spectrums.spectrum import Spectrum
from src.headers.header import Header


class Cube(FitsFile):
    """
    Encapsulates the methods specific to data cubes.
    """

    def __init__(self, data: Array3D, header: Header=None):
        """
        Initialize a Cube object.

        Parameters
        ----------
        data : Array3D
            The values of the Cube.
        header : Header, default=None
            The header of the Cube.
        """
        self.data = data
        self.header = header

    def __eq__(self, other):
        same_array = np.allclose(self.data, other.data, equal_nan=True)
        same_header = self.header == other.header
        return same_array and same_header

    def __getitem__(self, slices: tuple[slice]) -> Spectrum | Map | Cube:
        int_slices = [isinstance(slice_, int) for slice_ in slices]
        if int_slices.count(True) == 1:
            map_header = self.header.flatten(axis=int_slices.index(True))
            return Map(data=Array2D(self.data[slices]), header=map_header)
        elif int_slices.count(True) == 2:
            first_int_i = int_slices.index(True)
            map_header = self.header.flatten(axis=first_int_i)
            spectrum_header = map_header.flatten(axis=(int_slices.index(True, start=(first_int_i+1))))
            return Spectrum(data=self.data[slices], header=spectrum_header)
        elif int_slices.count(True) == 3:
            return self.data[slices]
        else:
            return self.__class__(self.data[slices], self.header.crop_axes(slices))
    
    def __iter__(self):
        self.iter_n = -1
        return self
    
    def __next__(self):
        self.iter_n += 1
        if self.iter_n >= self.data.shape[1]:
            raise StopIteration
        else:
            return self[:,self.iter_n,:]

    def copy(self):
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
        cube : Cube
            Loaded Cube.
        """
        fits_object = fits.open(filename)[0]
        cube = cls(
            Array3D(fits_object.data),
            Header(fits_object.header)
        )
        return cube

    def save(self, filename: str, overwrite: bool=False):
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

    def bin(self, bins: tuple[int, int, int], ignore_nans: bool=False) -> Cube:
        """
        Bins a Cube.

        Parameters
        ----------
        bins : tuple[int, int, int]
            Number of pixels to be binned together along each axis (1-3). A value of 1 results in the axis not being
            binned. The axes are in the order z, y, x.
        ignore_nans : bool, default=False
            Whether to ignore the nan values in the process of binning. If no nan values are present, this parameter is
            obsolete. If False, the function np.mean is used for binning whereas np.nanmean is used if True. If the nans
            are ignored, the cube might increase in size as pixels will take the place of nans. If the nans are not
            ignored, the cube might decrease in size as every new pixel that contains a nan will be made a nan also.

        Returns
        -------
        cube : Cube
            Binned Cube.
        """
        assert list(bins) == list(filter(lambda val: val >= 1 and isinstance(val, int), bins)), \
            f"{C.LIGHT_RED}All values in bins must be integers greater than or equal to 1.{C.END}"
        if ignore_nans:
            func = np.nanmean
        else:
            func = np.mean

        cropped_pixels = np.array(self.data.shape) % np.array(bins)
        data_copy = self.data[:self.data.shape[0] - cropped_pixels[0],
                              :self.data.shape[1] - cropped_pixels[1], 
                              :self.data.shape[2] - cropped_pixels[2]]

        for ax, b in enumerate(bins):
            if b != 1:
                indices = list(data_copy.shape)
                indices[ax:ax+1] = [data_copy.shape[ax] // b, b]
                reshaped_data = data_copy.reshape(indices)
                data_copy = func(reshaped_data, axis=ax+1)

        return self.__class__(data_copy, self.header.bin(bins))

    def invert_axis(self, axis: int) -> Cube:
        """
        Inverts the elements' order along an axis.

        Parameters
        ----------
        axis : int
            Axis whose order must be flipped. 0, 1, 2 correspond to z, y, x respectively.

        Returns
        -------
        cube : Cube
            Cube with the newly axis-flipped Data_cube.
        """
        return self.__class__(np.flip(self.data, axis=axis), self.header.invert_axis(axis))

    def swap_axes(self, axis_1: int, axis_2: int) -> Cube:
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
        cube : Cube
            Cube with the switched axes.
        """
        new_data = self.data.copy().swapaxes(axis_1, axis_2)
        new_header = self.header.swap_axes(axis_1, axis_2)
        return self.__class__(new_data, new_header)
