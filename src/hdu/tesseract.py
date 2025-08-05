from __future__ import annotations
import numpy as np
import awkward as ak
import astropy.units as u
from src.graphinglib import Curve
from typing import Self, Callable, Literal
from collections import namedtuple
from astropy.io import fits
from astropy.modeling.models import Gaussian1D
from colorist import BrightColor as C

from src.hdu.fits_file import FitsFile
from src.headers.header import Header
from src.hdu.maps.grouped_maps import GroupedMaps
from src.hdu.maps.map import Map
from src.hdu.arrays.array_2d import Array2D
from src.hdu.cubes.cube import Cube


class Tesseract(FitsFile):
    """
    This class implements a fit results manager and encapsulates methods specific to four dimensional inhomogeneous data
    management.

    .. note::
        In the following documentation, the models are assumed to be gaussian for simplicity, but this class can be
        used with any model and number of parameters.
    """

    def __init__(self, data: np.ndarray, header: Header):
        """
        Initializes a Tesseract object. The given data is assumed as such : along the first axis (axis=0) is the
        parameters of each gaussian used for fitting, along the second axis (axis=1) is every gaussian that was used for
        fitting a specific pixel and the two last axes represent the image itself (y and x respectively).
        Warning : the format of the given data is not the same as in the from_ak_array method.

        Parameters
        ----------
        data : np.ndarray
            The values of the Tesseract.
        header : Header
            The header of the Tesseract.

        Notes
        -----
        This means that the parameters at index [0,...] should be the amplitudes of each fitted gaussian, at index
        [1,...] the amplitude uncertainties, at index [2,...] the mean value of the gaussians, etc. The elements at
        index [:,0,...] are the amplitudes of the first gaussian, at index [:,1,...] the amplitudes of the second
        gaussian, etc. The final two axes (y and x) represent the image itself, so the data at index [:,:,y,x] is the
        parameters of all gaussians at pixel (y,x).
        """
        self.data = data
        self.header = header

    @classmethod
    def from_ak_array(cls, data: ak.Array, header: Header) -> Tesseract:
        """
        Initializes a Tesseract object. The given data is assumed as such : the first two axes represent the image
        itself (y and x respectively), along the third axis (axis=2) is the data of each gaussian that was used for
        fitting a specific pixel and along the fourth axis (axis=3) is the parameters of each gaussian used for fitting.

        .. warning:
            The format of the given data is not the same as in the constructor.

        Parameters
        ----------
        data : ak.Array
            The values of the Tesseract.
        header : Header
            The header of the Tesseract.

        Returns
        -------
        Tesseract
            An instance of the given class containing the given data and header.
        """
        return cls(cls.swapaxes(cls.rectangularize(data)), header)

    def __setitem__(self, key: tuple[slice | int], value: float | np.ndarray):
        """
        Sets the parameters of a certain fitted gaussian at a specified pixel. Can also be used to remove entirely a
        gaussian by setting it to np.nan.

        Parameters
        ----------
        key : tuple[slice | int]
            Three element tuple specifying where the values should be placed. The three elements refer respectively to
            the gaussian function number [0,N[ where N is the number of fitted gaussians in the Tesseract, the y
            coordinate and the x coordinate.
        value : float | np.ndarray
            Value to place at the given slice. If value is a float, then all the gaussian parameters for the specified
            gaussian function number will be set to this value. This is especially useful for setting a gaussian to NAN.
            If value is a np.ndarray, then it must contain a multiple of six elements which will attribute the six
            values to the gaussian parameters.
            Note : the setting of an individual gaussian at a time is encouraged to facilitate manipulations of
            dimensions.
        """
        assert len(key) == 3, f"{C.RED}key must have 3 elements; current length is {len(key)}.{C.OFF}"
        assert (isinstance(value, float) and np.isnan(value)) or value.shape == (6,), \
            f"{C.RED}value must be np.nan or a six elements array.{C.OFF}"
        if isinstance(value, float):
            self.data.__setitem__((slice(None,None),*key), np.full(6, value))
        else:
            self.data.__setitem__((slice(None,None),*key), value)

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
        Tesseract
            An instance of the given class containing the file's contents.
        """
        fits_object = fits.open(filename)[0]
        tesseract = cls(
            fits_object.data,
            Header(fits_object.header)
        )
        if len(tesseract.data.shape) != 4:
            raise TypeError(C.RED+"The provided file is not a Tesseract."+C.OFF)
        return tesseract

    @staticmethod
    def rectangularize(array: ak.Array) -> np.ndarray:
        """
        Rectangularizes the Tesseract object by padding the data along every inhomogeneous axis with np.nan.

        Parameters
        ----------
        array : ak.Array
            Inhomogeneous array to rectangularize.

        Returns
        -------
        np.ndarray
            Rectangularized array.
        """
        for i, value in zip(range(1,4), [[], [], np.nan]):
            len_ = ak.max(ak.num(array, axis=i))
            # Nones are replaced by lists so further padding can take place (along the axis=1)
            array = ak.pad_none(array, len_, axis=i)
            # The following Nones are replaced with np.nans
            array = ak.fill_none(array, value, axis=None)

        return array.to_numpy()

    @staticmethod
    def swapaxes(array: np.ndarray) -> np.ndarray:
        """
        Swaps the axes of an array as such : (y, x, gaussians, parameters) -> (parameters, gaussians, y, x).

        Parameters
        ----------
        array : np.ndarray
            Initial array to swap.

        Returns
        -------
        np.ndarray
            Swapped array.
        """
        swapped_array = array.swapaxes(0, 2).swapaxes(1, 3).swapaxes(0, 1)
        return swapped_array

    def save(self, filename: str, overwrite: bool = False):
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

    def get_spectrum_plot(self, cube: Cube, coords: tuple[int, int], model: Callable = None) -> tuple[Curve]:
        """
        Gives the spectrum and its fit at the given coordinates with two Curves.

        Parameters
        ----------
        cube : Cube
            Cube from which to get the spectrum data. This must be the cube with which the Tesseract was constructed.
        coords : tuple[int, int]
            Coordinates at which the Spectrum needs to be given.
        model : Callable, optional
            Model to be used for the fit. This must be a callable function that takes n + 1 parameters, where n is the
            number of parameters of the model. If None, the model will be assumed to be a Gaussian.

        Returns
        -------
        tuple[Curve]
            The first element is the Spectrum's data at every channel and the last element is the Spectrum's total fit.
            The middle elements are every individual gaussian fitted. If a single Gaussian was fitted, the y_data of the
            middle and last Curves will be identical.
        """
        if model is None:
            model = lambda x, *params: Gaussian1D(*params)(x)
        spectrum = cube[:, *coords]

        # Isolate amplitudes, means and stddevs at the given coords
        params = self.data[::2,:,coords[0],coords[1]].transpose()
        spectrum_individual_models = []
        for i, params_i in enumerate(params):
            if not np.all(np.isnan(params_i)):
                spectrum_individual_models.append(
                    Curve(
                        spectrum.x_values,
                        model(spectrum.x_values, *params_i),
                        label=f"Model {i}"
                    )
                )

        if not spectrum_individual_models:
            print("There is no successful fit at the given coordinates.")
            return spectrum.plot,
        else:
            spectrum_total = sum(spectrum_individual_models)
            spectrum_total.label = "Sum"
            return spectrum.plot, *spectrum_individual_models, spectrum_total

    def filter(self, slice: slice) -> Self:
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
        Self
            Tesseract object with the specified slice applied to the third element of the first axis.
        """
        i = 2
        # Make a boolean filter to determine the axes that fulfill the condition
        filter_3d = (((slice.start if slice.start else 0) <= self.data[i,:,:,:]) &
                     (self.data[i,:,:,:] < (slice.stop if slice.stop else 1e5)))
        filter_4d = np.tile(filter_3d, (self.data.shape[0], 1, 1, 1))
        # Convert boolean filter to True/np.nan
        filter_4d = np.where(filter_4d, filter_4d, np.nan)
        return self.__class__(self.data * filter_4d, self.header)

    def to_grouped_maps(self, names: list[str] = ["amplitude", "mean", "stddev"]) -> GroupedMaps:
        """
        Converts the Tesseract object into a GroupedMaps object.

        Parameters
        ----------
        names : list[str], default=["amplitude", "mean", "stddev"]
            Names of the maps to be created from the Tesseract, in the order the parameters are stored in the Tesseract.

        Returns
        -------
        GroupedMaps
            A GroupedMaps object containing the Tesseract's parameters as Maps. If the Tesseract contains gaussian data,
            the object gives the amplitude, mean, and standard deviation maps extracted from the Tesseract.
        """
        if len(names) != int(self.data.shape[0] / 2):
            raise ValueError(
                f"{C.RED}The number of names ({len(names)}) must be equal to half the number of parameters in the "
                f"Tesseract ({self.data.shape[0]}).{C.OFF}"
            )
        maps = namedtuple("maps", names)
        maps = maps(*[[] for _ in names])

        new_header = self.header.celestial

        for i, name in zip(range(0, self.data.shape[0], 2), names):
            for j in range(self.data.shape[1]):
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

    def split(self, indice: int, axis: Literal[2, 3]) -> list[Self, Self]:
        """
        Splits a Tesseract into two smaller Tesseracts along a certain axis at a given indice.

        Parameters
        ----------
        indice : int
            Indice of where the Tesseract will be splitted. For example, indice=10 will give a Tesseract that goes from
            0 to 9 and another from 10 to ... along a given axis.
        axis : Literal[2, 3]
            Axis along which to split the Tesseract. This should be 2 or 3 to split in the coordinate axes.

        Returns
        -------
        list[Self, Self]
            Tesseracts that were splitted. The list is ordered so the Tesseract with the lowest slice is given first.
        """
        if axis not in (2, 3):
            raise ValueError(f"{C.RED}Axis must be 2 or 3, got {axis}.{C.OFF}")

        splitted = np.split(self.data, [indice], axis)
        slices = [slice(None, indice), slice(indice, None)]
        header_slices = [[slice(None), slice(None), slice(None), slice(None)] for _ in range(2)]    # this forces copy
        header_slices[0][axis] = slices[0]
        header_slices[1][axis] = slices[1]

        tess = [self.__class__(data, self.header.slice(h_slice)) for data, h_slice in zip(splitted, header_slices)]
        return tess

    def concatenate(self, other, axis: int) -> Self:
        """
        Concatenates two Tesseracts into a single Tesseract. The Tesseract closest to the origin must be the one to call
        this method as the second Tesseract is added to the right (axis=3)/up (axis=2) depending on the chosen axis.

        Parameters
        ----------
        other : Tesseract
            Tesseract to be merged.
        axis : int
            Axis along which to merge the Tesseract. This should be 2 or 3 to merge in the coordinate axes.

        Returns
        -------
        Self
            Merged Tesseract along the specified axis.
        """
        new_data = np.concatenate((self.data, other.data), axis)
        new_header = self.header.concatenate(other.header, axis)
        return self.__class__(new_data, new_header)

    def compress(self) -> Self:
        """
        Compresses the nan values of a Tesseract and removes unnecessary slices.

        Returns
        -------
        Self
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
                constant_values=np.nan
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
