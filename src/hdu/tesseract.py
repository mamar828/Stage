from __future__ import annotations
import numpy as np
import awkward as ak
from collections import namedtuple
from astropy.io import fits

from src.hdu.fits_file import FitsFile
from src.headers.header import Header
from src.hdu.maps.grouped_maps import GroupedMaps
from src.hdu.maps.map import Map
from src.hdu.arrays.array_2d import Array2D


class Tesseract(FitsFile):
    """
    Encapsulates the methods specific to four dimensional inhomogeneous data management.
    """

    def __init__(self, data: np.ndarray, header: Header):
        """
        Initializes a Tesseract object. The given data is assumed as such : along the first axis (axis=0) is the
        parameters of each gaussian used for fitting, along the second axis (axis=1) is the data of each gaussian that
        was used for fitting a specific pixel and the two last axes represent the image itself (y and x respectively).
        Warning : the format of the given data is not the same as in the from_ak_array method.

        Parameters
        ----------
        data : np.ndarray
            The values of the Tesseract.
        header : Header
            The header of the Tesseract.
        """
        self.data = data
        self.header = header

    @classmethod
    def from_ak_array(cls, data: ak.Array, header: Header) -> Tesseract:
        """
        Initializes a Tesseract object. The given data is assumed as such : the first two axes represent the image
        itself (y and x respectively), along the third axis (axis=2) is the data of each gaussian that was used for
        fitting a specific pixel and along the fourth axis (axis=3) is the parameters of each gaussian used for fitting.
        Warning : the format of the given data is not the same as in the constructor.

        Parameters
        ----------
        data : ak.Array
            The values of the Tesseract.
        header : Header
            The header of the Tesseract.
        
        Returns
        -------
        tesseract : Tesseract
            Initialized Tesseract.
        """
        return cls(cls.swapaxes(cls.rectangularize(data)), header)

    @classmethod
    def load(cls, filename: str) -> Tesseract:
        """
        Loads a Tesseract from a .fits file.

        Parameters
        ----------
        filename : str
            Name of the file to load.

        Returns
        -------
        tesseract : Tesseract
            Loaded Tesseract.
        """
        fits_object = fits.open(filename)[0]
        tesseract = cls(
            fits_object.data,
            Header(fits_object.header)
        )
        return tesseract

    @staticmethod
    def rectangularize(array: ak.Array) -> np.ndarray:
        """
        Rectangularizes the Tesseract object by padding the data along every inhomogeneous axis with np.NAN.

        Parameters
        ----------
        array : ak.Array
            Inhomogeneous array to rectangularize.
        
        Returns
        -------
        array : np.ndarray
            Rectangularized array.
        """
        for i, value in zip(range(1,4), [[], [], np.NAN]):
            len_ = ak.max(ak.num(array, axis=i))
            array = ak.pad_none(array, len_, axis=i)
            array = ak.fill_none(array, value, axis=None)

        # Nones are replaced by lists so further padding can take place (along the axis=1)
        # The following Nones are replaced with np.NANs
        return array.to_numpy()

    @staticmethod
    def swapaxes(array: np.ndarray) -> np.ndarray:
        """
        Swaps the axes of the Tesseract object as such : (y, x, gaussians, parameters) -> (parameters, gaussians, y, x).

        Parameters
        ----------
        array : ak.Array
            Initial array to swap.
        
        Returns
        -------
        array : np.ndarray
            Swapped array.
        """
        swapped_array = array.swapaxes(0, 2).swapaxes(1, 3).swapaxes(0, 1)
        return swapped_array
    
    def save(self, filename: str, overwrite: bool=False):
        """
        Saves a Tesseract to a file.

        Parameters
        ----------
        filename : str
            Filename in which to save the Tesseract.
        overwrite : bool, default=False
            Whether the file should be forcefully overwritten if it already exists.
        """
        hdu_list = fits.HDUList([fits.PrimaryHDU(self.data, self.header)])
        super().save(filename, hdu_list, overwrite)

    def filter(self, slice: slice) -> Tesseract:
        """
        Filters the Tesseract to get only the elements whose third value along the first axis is between the specified
        slice. In the case of the Tesseract outputted by the src.hdu.cubes.cube_co.CubeCO.fit method, The returned
        Tesseract will contain only the gaussians whose mean value was contained in the given slice.

        Parameters
        ----------
        slice : slice
            The slice to be applied to the Tesseract. Each element of axis=0 will be outputted if the third value
            (index=2) of that axis fits in the slice.

        Returns
        -------
        filtered tesseract : Tesseract
            Tesseract object with the specified slice applied to the third element of the first axis.
        """
        i = 2
        # Make a boolean filter to determine the axes that fulfill the condition
        filter_3d = (((slice.start if slice.start else 0) <= self.data[i,:,:,:]) & 
                     (self.data[i,:,:,:] < (slice.stop if slice.stop else 1e5)))
        filter_4d = np.tile(filter_3d, (6, 1, 1, 1))
        # Convert boolean filter to True/np.NAN
        filter_4d = np.where(filter_4d, filter_4d, np.NAN)
        return self.__class__(self.data * filter_4d, self.header)

    def to_grouped_maps(self) -> GroupedMaps:
        """
        Converts the Tesseract object into a GroupedMaps object.

        Returns
        -------
        maps : GroupedMaps
            A GroupedMaps object containing the amplitude, mean, and standard deviation maps extracted from the
            Tesseract.
        """
        names = ["amplitude", "mean", "stddev"]
        maps = namedtuple("maps", names)
        maps = maps([], [], [])

        new_header = self.header.copy()
        for i in range(self.header["NAXIS"] - 2):
            new_header = self.header.flatten(axis=0)

        for i, name in zip(range(0, 6, 2), names):
            for j in range(np.shape(self.data)[1]):
                getattr(maps, name).append(
                    Map(
                        data=Array2D(self.data[i,j,:,:]),
                        uncertainties=Array2D(self.data[i+1,j,:,:]),
                        header=new_header
                    )
                )

        gm = GroupedMaps(
            maps=[(name, getattr(maps, name)) for name in names]
        )
        return gm

    def split(self, indice: int, axis: int) -> list[Tesseract, Tesseract]:
        """
        Splits a Tesseract into two smaller Tesseracts along a certain axis at a given indice.
        
        Parameters
        ----------
        indice : int
            Indice of where the Tesseract will be splitted. For example, indice=10 will give a Tesseract that goes from
            0 to 9 and another from 10 to ... along a given axis.
        axis : int
            Axis along which to split the Tesseract. This should be 0 or 1 to split in the coordinate axes.

        Returns
        -------
        splitted tesseracts : list[Tesseract, Tesseract]
            Tesseracts that were splitted. The list is ordered so the Tesseract with the lowest slice is given first.
        """
        splitted = np.split(self.data, [indice], axis)

        slices = [slice(None, indice), slice(indice, None)]
        header_slices = [[0, 0, 0] for _ in slices]
        [header_slices[i].insert(axis, slice_) for i, slice_ in enumerate(slices)]
        
        tess = [self.__class__(data, self.header.crop_axes(h_slice)) for data, h_slice in zip(splitted, header_slices)]
        return tess

    def concatenate(self, other, axis: int) -> Tesseract:
        """
        Concatenates two Tesseracts into a single Tesseract. The Tesseract closest to the origin must be the one to call
        this method as the second Tesseract is added to the right (axis=3)/up (axis=2) depending on the chosen axis.

        Parameters
        ----------
        other : Tesseract
            Tesseract to be merged.
        axis : int
            Axis along which to merge the Tesseract. This should be 0 or 1 to merge in the coordinate axes.

        Returns
        -------
        merged tesseract : Tesseract
            Merged Tesseract along the specified axis.
        """
        new_data = np.concatenate((self.data, other.data), axis)
        new_header = self.header.concatenate(other.header, axis)
        return self.__class__(new_data, new_header)

    def compress(self) -> Tesseract:
        """
        Compresses the nan values of a Tesseract and removes unnecessary slices.

        Returns
        -------
        compressed tesseract : Tesseract
            Tesseract with the gaussian function indices shifted to the uppermost level and with all nan slices removed.
        """
        def collapse(row):
            # Collapses the elements of a row to place the non nan elements at the first index and pad with the same
            # number of nan values
            elements = row[~np.isnan(row)]
            padded = np.pad(
                elements,
                pad_width=(0, row.shape[0] - elements.shape[0]),
                mode="constant",
                constant_values=np.NAN
            )
            return padded
        
        # Collapse the non nan elements at the first indices
        collapsed_array = np.apply_along_axis(collapse, axis=1, arr=self.data)
        # collapsed_array is now an array with the same shape as before, but with every gaussian function shifted to the
        # upermost position (first gaussian index, or later if there is more than one gaussian function at that pixel)

        # Find the gaussian function indices where all parameters are None (all parameters : axis=0, y,x : axis=(2,3))
        nan_indexes = np.all(np.isnan(collapsed_array), axis=(0,2,3))
        # nan_indexes is now a 1D array giving which indice is always None

        # Filter the rows to keep only the data where there is not always nan parameters for all the slice
        filtered_rows = collapsed_array[:,~nan_indexes,:,:]
        
        return self.__class__(filtered_rows, self.header)
