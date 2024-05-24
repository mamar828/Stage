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

    def __init__(self, data: ak.Array, header: Header=None):
        """
        Initializes a Tesseract object.

        Parameters
        ----------
        data : ak.Array
            The values of the Tesseract.
        header : Header, default=None
            The header of the Tesseract.
        """
        self.data = data
        header["CTYPE3"] = "amplitude + unc., mean + unc., stddev + unc."
        header["CTYPE4"] = "gaussian function index"
        self.header = header
    
    @property
    def is_rectangular(self) -> bool:
        """
        Checks if the Tesseract is rectangular.

        Returns
        -------
        is_rectangular : bool
            True if the Tesseract is rectangular, False otherwise.
        """
        # Perform a check only on axis=2 and axis=3 as the first two axes are always rectangular
        len_2 = ak.max(ak.num(self.data, axis=2))
        axis_2 = ak.all(ak.num(self.data, axis=2) == len_2)
        len_3 = ak.max(ak.num(self.data, axis=3))
        axis_3 = ak.all(ak.num(self.data, axis=3) == len_3)
        return axis_2 and axis_3
    
    @staticmethod
    def requires_rectangular(func):
        # Decorator to automatically rectangularize the Tesseract when needed
        def inner_func(self, *args, **kwargs):
            if not self.is_rectangular:
                self.rectangularize()
            return func(self, *args, **kwargs)
        return inner_func
    
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
        data = np.swapaxes(np.swapaxes(fits_object.data, 0, 2), 1, 3)
        ak_data = ak.Array(data.tolist())
        header = Header(fits_object.header)
        tesseract = cls(
            ak_data,
            header
        )
        return tesseract

    @requires_rectangular
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
        new_header = self.header.copy()
        new_data = np.swapaxes(np.swapaxes(self.data.to_numpy(), 0, 2), 1, 3)
        hdu_list = fits.HDUList([fits.PrimaryHDU(new_data, self.header)])
        super().save(filename, hdu_list, overwrite)

    @requires_rectangular
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
        if not self.is_rectangular:
            self.rectangularize

        # Make a boolean filter to determine the axes that fulfill the condition
        filter_ = (((slice.start if slice.start else 0) <= self.data[:,:,:,2]) & 
                   (self.data[:,:,:,2] < (slice.stop if slice.stop else 1e5)))

        return Tesseract(self.data[filter_], self.header)

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

    @requires_rectangular
    def to_grouped_maps(self) -> GroupedMaps:
        """
        Converts the Tesseract object into a GroupedMaps object.

        Returns
        -------
        maps : GroupedMaps
            A GroupedMaps object containing the amplitude, mean, and standard deviation maps extracted from the
            Tesseract.
        """
        number_of_maps = ak.max(ak.num(self.data, axis=2))

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
