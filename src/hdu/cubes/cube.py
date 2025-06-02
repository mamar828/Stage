from __future__ import annotations
import numpy as np
import scipy as sp
import pyregion
from astropy.io import fits
from typing import Self, Any, Literal
from colorist import BrightColor as C
from pathos.pools import ProcessPool
from pathos.helpers import cpu_count
from copy import deepcopy
from tqdm import tqdm

from src.hdu.fits_file import FitsFile
from src.hdu.arrays.array_2d import Array2D
from src.hdu.arrays.array_3d import Array3D
from src.hdu.maps.map import Map, MapCO
from src.spectrums.spectrum import Spectrum
from src.spectrums.spectrum_co import SpectrumCO
from src.headers.header import Header
from src.base_objects.silent_none import SilentNone
from src.tools.array_functions import list_to_array
from src.tools.messaging import notify_function_end


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
            raise TypeError(f"{C.RED}Every slice element must be an int or a slice.{C.OFF}")
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

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    def copy(self) -> Self:
        return self.__class__(deepcopy(self.data), deepcopy(self.header))
    
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
            Cube with the newly axis-flipped data.
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

    @FitsFile.silence_function
    def get_masked_region(self, region: pyregion.core.ShapeList) -> Self:
        """
        Gives the Cube within a region.

        Parameters
        ----------
        region : pyregion.core.ShapeList
            Region that will be kept in the final Cube. If None, the whole cube is returned.
        
        Returns
        -------
        Self
            Masked Cube.
        """
        if region:
            if self.header:
                mask = region.get_mask(self[0,:,:].data.get_PrimaryHDU(self.header))
            else:
                mask = region.get_mask(shape=self.data.shape[1:])
            mask = np.where(mask == False, np.nan, 1)
            mask = np.tile(mask, (self.data.shape[0], 1, 1))
        else:
            mask = np.ones_like(self.data)
        return self.__class__(
            self.data * mask,
            self.header
        )

    def find_peaks_gaussian_estimates(self, voigt: bool = False, **kwargs) -> Self:
        """
        Finds gaussian initial guesses using a find_peaks algorithm. These initial guesses can then be used to fit
        gaussian functions to the data.

        Parameters
        ----------
        voigt : bool, default=False
            If True, the initial guesses will be made for a Voigt profile instead of a Gaussian profile. This very 
            simple option simply duplicates the stddev parameter to give four parameters for the Voigt profile.

            .. note::
                Voigt profiles are typically defined by the FWHM of the Lorentzian and Gaussian components, but this
                method uses the standard deviation of the Gaussian component as a proxy for both components. This is a
                very rough approximation and may not yield accurate results for all data.

        kwargs : Any
            Arguments to pass to the scipy.signal.find_peaks function. Useful parameters include:
            - `height`: Required height of the peaks.
            - `threshold`: Required threshold of peaks, the vertical distance to its neighboring samples.
            - `distance`: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
            - `prominence`: Required prominence of peaks.
            - `width`: Required width of peaks in samples.
            See the documentation of `scipy.signal.find_peaks` for more details.

        Returns
        -------
        Self
            Cube with the initial guesses for the Gaussian model. The guesses are stored along the first axis, ordered
            as: amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, ..., where the first three values are the
            parameters of the first Gaussian model, the next three are the parameters of the second Gaussian model, and
            so on.
        """
        transposed_data = Cube.flatten_3d_array(self.data)
        peak_means = [sp.signal.find_peaks(spectrum, **kwargs)[0] for spectrum in transposed_data]
        peak_amplitudes = [spectrum[peaks] for spectrum, peaks in zip(transposed_data, peak_means)]
        peak_means = list_to_array(peak_means)
        peak_amplitudes = list_to_array(peak_amplitudes)
        assert peak_means.size > 0, \
            "No peaks were detected in the data. Please check the parameters passed to find_peaks."

        # Estimate stddevs
        peak_stddevs = []
        for means, amplitude in zip(peak_means.T, peak_amplitudes.T):    # iterate over each detected peak
            half_max_difference = transposed_data - amplitude[:,None] / 2
            half_max_intersect_mask = np.abs(np.diff(np.sign(half_max_difference))).astype(bool)
            intersects_x = [np.where(mask)[0] + 1 for mask in half_max_intersect_mask]

            current_stddevs = []
            for intersect, mean in zip(intersects_x, means):
                if np.isnan(mean):
                    current_stddevs.append(np.nan)
                else:
                    lower_bound_candidates = intersect[intersect < mean]
                    lower_bound = lower_bound_candidates.max() if len(lower_bound_candidates) > 0 else 0
                    upper_bound_candidates = intersect[intersect > mean]
                    upper_bound = upper_bound_candidates.min() if len(upper_bound_candidates) > 0 else 0
                    current_stddevs.append((upper_bound - lower_bound) / (2*np.sqrt(2*np.log(2))))

            peak_stddevs.append(current_stddevs)

        peak_stddevs = np.array(peak_stddevs).T
        peak_means += 1     # correct for the 0-based indexing in numpy but 1-based indexing in the data

        # Combine the results into a single array and reshape it
        if voigt:
            guesses = np.dstack((peak_amplitudes, peak_means, peak_stddevs, peak_stddevs))
        else:
            guesses = np.dstack((peak_amplitudes, peak_means, peak_stddevs))        # shape is (n_data, n_models, 3)
            
        guesses = guesses.reshape(self.data.shape[2], self.data.shape[1], -1)
        guesses = guesses.T

        return self.__class__(guesses, self.header)

    @notify_function_end
    def fit(self, model, guesses: Cube | Array3D, number_of_tasks: int | Literal["auto"] = "auto", **kwargs) -> Self:
        """
        Fits a model to the Cube data. This function wraps the `scipy.optimize.curve_fit` function and for an entire 
        Cube, and uses multiprocessing to speed up the fitting process.

        Parameters
        ----------
        model : callable
            The model to fit to the data. This must be a callable function with the signature:
            `model(x, *params)`, where `x` is the independent variable and `params` are the parameters to fit. The
            number of parameters must match number of parameters given in `guesses`.
        guesses : Cube | Array3D
            Initial guesses for the parameters of the model. If None, the function will try to find initial guesses. The
            guesses must be given along the first axis, ordered as:
            amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, ..., where the first three values are the
            parameters of the first Gaussian model, the next three are the parameters of the second Gaussian model, and
            so on. The output of the `find_peaks_gaussian_estimates` method can be used as is.
        number_of_tasks : int | Literal["auto"], default="auto"
            Number of tasks to split the fitting process into. If "auto", it will be set to the number of CPU cores
            available on the system.
        kwargs : Any
            Additional arguments to pass to the fitting function.

        Returns
        -------
        Self
            Cube with fitted models. The fitted parameters are stored identically to the guesses, i.e. every group of
            three parameters along the first axis corresponds to a single model, ordered as amplitude, mean and stddev.
        """
        guesses_array = guesses.data if isinstance(guesses, Cube) else guesses
        if number_of_tasks == "auto":
            number_of_tasks = cpu_count()
        x_values = np.arange(self.shape[0]) + 1

        @FitsFile.silence_function
        def worker_fit_spectrums(spectrums, guesses):
            results = []
            for spectrum_i, guesses_i in zip(spectrums, guesses):
                # Filter out invalid guesses (rows with np.nan)
                valid_guesses = guesses_i[~np.isnan(guesses_i)]
                if valid_guesses.size == 0:
                    params = np.full(guesses_i.size, np.nan)
                else:
                    # Flatten valid guesses and fit
                    try:
                        params = sp.optimize.curve_fit(
                            f=model,
                            xdata=x_values,
                            ydata=spectrum_i,
                            p0=valid_guesses.flatten(),
                            maxfev=kwargs.get("maxfev", 10000),
                        )[0]
                    except RuntimeError:
                        params = np.full(guesses_i.size, np.nan)

                # Reshape to match the original guesses' shape
                result = np.full(guesses_i.size, np.nan)
                result[:params.size] = params
                results.append(result)
            return results

        data_2d, guesses_2d = self.flatten_3d_array(self.data), self.flatten_3d_array(guesses_array)
        splitted_data = np.array_split(data_2d, number_of_tasks)
        splitted_guesses = np.array_split(guesses_2d, number_of_tasks)
        packed_arguments = [(chunk_data, chunk_guesses) 
                            for chunk_data, chunk_guesses in zip(splitted_data, splitted_guesses)]

        fit_params_chunks = []
        pbar = tqdm(total=len(packed_arguments), desc="Fitting", unit="chunk", colour="blue", miniters=1)
        with ProcessPool() as pool:
            for result in pool.imap(lambda args: worker_fit_spectrums(*args), packed_arguments):
                fit_params_chunks.append(result)
                pbar.update(1)

        fit_params = np.concatenate(fit_params_chunks, axis=0).reshape(self.data.shape[2], self.data.shape[1], -1).T

        return self.__class__(fit_params, self.header)

    @staticmethod
    def flatten_3d_array(array_3d: Array3D) -> Array2D:
        """
        Flattens a 3D array into a 2D array by transposing the array and reshaping it by combining the first two axes.

        Parameters
        ----------
        array_3d : Array3D
            The 3D array to flatten.

        Returns
        -------
        Array2D
            The flattened 2D array, which kept the spectral axis intact and combined the spatial axes.
        """
        return Array2D(array_3d.T.reshape(array_3d.shape[2] * array_3d.shape[1], array_3d.shape[0]))
