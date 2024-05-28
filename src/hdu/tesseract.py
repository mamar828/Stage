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

    def __init__(self, data: ak.Array | np.ndarray, header: Header):
        """
        Initializes a Tesseract object. The given data is assumed as such : the first two axes represent the image
        itself (y and x respectively), along the third axis (axis=2) is the data of each gaussian that was used for
        fitting a specific pixel and along the fourth axis (axis=3) is the parameters of each gaussian used for fitting.

        Parameters
        ----------
        data : ak.Array | np.ndarray
            The values of the Tesseract.
        header : Header
            The header of the Tesseract.
        """
        self.data = data
        self.header = header.copy()
        self.header["CTYPE3"] = "amplitude + unc., mean + unc., stddev + unc."
        self.header["CTYPE4"] = "gaussian function index"
        if not self.is_rectangular:
            self.rectangularize()

    @property
    def is_rectangular(self) -> bool:
        """
        Checks if the Tesseract is rectangular.

        Returns
        -------
        is_rectangular : bool
            True if the Tesseract is rectangular, False otherwise.
        """
        if isinstance(self.data, np.ndarray):
            return True
        
        # Perform a check only on axis=2 and axis=3 as the first two axes are always rectangular
        len_2 = ak.max(ak.num(self.data, axis=2))
        axis_2 = ak.all(ak.num(self.data, axis=2) == len_2)
        len_3 = ak.max(ak.num(self.data, axis=3))
        axis_3 = ak.all(ak.num(self.data, axis=3) == len_3)
        return axis_2 and axis_3
    
    def rectangularize(self):
        """
        Rectangularizes the Tesseract object by padding the data along axis=2 and axis=3 with np.NAN. The first two axes
        are considered always rectangular (y, x).
        """
        len_2 = ak.max(ak.num(self.data, axis=2))
        padded_2 = ak.pad_none(self.data, len_2, axis=2)
        # Replace None by lists so further padding can take place (along the axis=3)
        padded_2 = ak.fill_none(padded_2, [], axis=None)

        len_3 = ak.max(ak.num(self.data, axis=3))
        padded_3 = ak.pad_none(padded_2, len_3, axis=3)

        self.data = ak.fill_none(padded_3, np.NAN, axis=None)

    @staticmethod
    def get_fits_data(data: np.ndarray) -> np.ndarray:
        """
        Swaps the given data to match the FITS file format (last axes are y, x).
    
        Parameters
        ----------
        data : np.ndarray
            The input array to be transformed.
    
        Returns
        -------
        swapped_data : np.ndarray
            The transformed numpy array with axes swapped to match the FITS file format.
        """
        swapped_data = np.swapaxes(np.swapaxes(data, 0, 2), 1, 3)
        return swapped_data
    
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
        data = Tesseract.get_fits_data(fits_object.data)
        header = Header(fits_object.header)
        tesseract = cls(
            data,
            header
        )
        return tesseract

    def save(self, filename: str, overwrite=False):
        """
        Saves a Tesseract to a file.

        Parameters
        ----------
        filename : str
            Filename in which to save the Tesseract.
        overwrite : bool, default=False
            Whether the file should be forcefully overwritten if it already exists.
        """
        new_data = Tesseract.get_fits_data(self.data)
        hdu_list = fits.HDUList([fits.PrimaryHDU(new_data, self.header)])
        super().save(filename, hdu_list, overwrite)

    def __getitem__(self, slice: slice) -> Tesseract:
        """
        Returns a new Tesseract object with the specified slice, which corresponds to the Tesseract's values along the 
        third element of the axis=3. The Tesseract is rectangularized

        Parameters
        ----------
        slice : slice
            The slice to be applied to the Tesseract. Each element of axis=2 will be outputted if the third value of
            that axis fits in the slice.

        Returns
        -------
        filtered tesseract : Tesseract
            Tesseract object with the specified slice applied to the third element of axis=3. The object is
            rectangularized.
        """
        # Make a boolean filter to determine the axes that fulfill the condition
        filter_ = (((slice.start if slice.start else 0) <= self.data[:,:,:,2]) & 
                   (self.data[:,:,:,2] < (slice.stop if slice.stop else 1e5)))

        return Tesseract(self.data[filter_], self.header)


    def to_grouped_maps(self) -> GroupedMaps:
        """
        Converts the Tesseract object into a GroupedMaps object.

        Returns
        -------
        maps : GroupedMaps
            A GroupedMaps object containing the amplitude, mean, and standard deviation maps extracted from the
            Tesseract.
        """
        number_of_maps = np.shape(self.data)[2]

        names = ["amplitude", "mean", "stddev"]
        maps = namedtuple("maps", names)
        maps = maps([], [], [])

        new_header = self.header.copy()
        for i in range(self.header["NAXIS"] - 2):
            new_header = self.header.flatten(axis=0)

        for i in range(number_of_maps):
            for j, name in zip(range(0, 6, 2), names):
                getattr(maps, name).append(
                    Map(
                        data=Array2D(self.data[:,:,i,j]),
                        uncertainties=Array2D(self.data[:,:,i,j+1]),
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
            0 to 9 and another from 10 to ... in a given axis.
        axis : int
            Axis along which to split the Tesseract. This should be 0 or 1 to split in the coordinate axes.

        Returns
        -------
        splitted tesseracts : list[Tesseract, Tesseract]
            Tesseracts that were splitted. The list is ordered so the Tesseract with the lowest slice is given first.
        """
        splitted_data = np.split(self.data, [0, indice], axis)[1:]      # Removes the first slice from 0-0
        slices = [slice(0, indice), slice(indice, None)]
        tesseracts = []
        for data, slice_ in zip(splitted_data, slices):
            h_axis = 2 - axis
            new_header = self.header.copy()
            stop = slice_.stop if slice_.stop is not None else self.header[f"NAXIS{h_axis}"]
            new_header[f"CRPIX{h_axis}"] -= slice_.start
            new_header[f"NAXIS{h_axis}"] = stop - slice_.start
            tesseracts.append(Tesseract(data, new_header.copy()))

        return tesseracts

    def merge(self, other, axis: int) -> Tesseract:
        """
        Merges two Tesseracts into a single Tesseract. The Tesseract that was splitted with the lowest indice must be
        the one to call this method.

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
        new_header = self.header.copy()
        h_axis = 2 - axis
        new_header[f"NAXIS{h_axis}"] += other.header.copy()[f"NAXIS{h_axis}"]
        return Tesseract(new_data, new_header)
