from __future__ import annotations
import numpy as np
from astropy.io import fits
from eztcolors import Colors as C

from src.hdu.fits_file import FitsFile
from src.hdu.arrays.array_3d import Array3D
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
        return np.array_equal(self.data, other.data) and self.header == other.header

    def __getitem__(self, slices: slice):
        """ Warning : indexing must be given in the order z, y, x. """
        return self.__class__(self.data[slices], self.header.get_cropped_axes(slices))
    
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

    def save(self, filename, overwrite=False):
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

    def bin(self, bins: tuple[int, int, int]) -> Cube:
        """
        Bins a Cube.

        Parameters
        ----------
        bins : tuple[int, int, int]
            Number of pixels to be binned together along each axis (1-3). A value of 1 results in the axis not being
            binned. The axes are in the order z, y, x.

        Returns
        -------
        cube : Cube
            Binned Cube.
        """
        assert list(bins) == list(filter(lambda val: val >= 1 and isinstance(val, int), bins)), \
            f"{C.LIGHT_RED}All values in bins must be greater than or equal to 1 and must be integers.{C.END}"

        cropped_pixels = np.array(self.data.shape) % np.array(bins)
        data_copy = self.data[:self.data.shape[0] - cropped_pixels[0],
                              :self.data.shape[1] - cropped_pixels[1], 
                              :self.data.shape[2] - cropped_pixels[2]]

        for ax, b in enumerate(bins):
            if b != 1:
                indices = list(data_copy.shape)
                indices[ax] = [data_copy.shape[ax] // b, b]
                # indices[ax:ax+1] = [int(data_copy.shape[ax]/b), b]
                reshaped_data = data_copy.reshape(indices)
                data_copy = reshaped_data.mean(axis=ax+1)

        return self.__class__(data_copy, self.header.bin(bins))

    def invert_axis(self, axis: int) -> Cube:
        """
        Inverts the elements' order along an axis.

        Arguments
        ---------
        axis : int
            Axis whose order must be flipped. 0, 1, 2 correspond to z, y, x respectively.

        Returns
        -------
        cube : Cube
            Cube with the newly axis-flipped Data_cube.
        """
        return self.__class__(np.flip(self.data, axis=axis), self.header.get_inverted_axis(axis))
