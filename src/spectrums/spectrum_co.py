from __future__ import annotations
import numpy as np
from matplotlib.axes import Axes
from scipy.constants import c
from scipy.signal import find_peaks
from astropy.modeling import models
from astropy import units as u

from src.spectrums.spectrum import Spectrum
from src.headers.header import Header


class SpectrumCO(Spectrum):
    """
    Encapsulates the methods specific to CO spectrums.
    """

    def __init__(
            self,
            data: np.ndarray,
            header: Header,
            peak_prominence: float=0.7,
            peak_minimum_height_sigmas: float=6.0,
            peak_minimum_distance: int=10,
            noise_channels: slice=slice(0,100)
        ):
        """
        Initializes a SpectrumCO object with a certain header, whose spectral information will be taken.

        Parameters
        ----------
        data : np.ndarray
            Detected intensity at each channel.
        header : Header
            Allows for the calculation of the FWHM using the header's informations.
        peak_prominence : float, default=0.7
            Required prominence of peaks to be detected as such. This is used in the scipy.signal.find_peaks function.
        peak_minimum_height_sigmas : float, default=6.0
            Minimum number of sigmas (stddev) above the continuum to be considered as a peak. This is used in the
            scipy.signal.find_peak function.
        peak_minimum_distance : int, default=10
            Minimum horizontal distance between peaks, in channels. This is used in the scipy.signal.find_peak function.
        noise_channels : slice, default=slice(0,100)
            Channels used to measure the noise's stddev. No peaks should be found in this region. 
        """
        super().__init__(data, header)
        self.PEAK_PROMINENCE = peak_prominence
        self._PEAK_MINIMUM_HEIGHT_SIGMAS = peak_minimum_height_sigmas
        self.PEAK_MINIMUM_DISTANCE = peak_minimum_distance
        self._NOISE_CHANNELS = noise_channels
        self._y_threshold = np.std(self.data[self._NOISE_CHANNELS]) * self._PEAK_MINIMUM_HEIGHT_SIGMAS

    @property
    def PEAK_MINIMUM_HEIGHT_SIGMAS(self, value: float):
        self._PEAK_MINIMUM_HEIGHT_SIGMAS = value
        self._y_threshold = np.std(self.data[self._NOISE_CHANNELS]) * self._PEAK_MINIMUM_HEIGHT_SIGMAS

    @property
    def NOISECHANNELS(self, value: slice):
        self._NOISECHANNELS = value
        self._y_threshold = np.std(self.data[self._NOISE_CHANNELS]) * self._PEAK_MINIMUM_HEIGHT_SIGMAS

    def plot_fit(self, ax: Axes, plot_all: bool=False, plot_initial_guesses: bool=False):
        """
        Sends the fitted functions to the plot() method to be plotted on an axis.

        Parameters
        ----------
        ax : Axes
            Axis on which to plot the Spectrum.
        plot_all : bool, default=False
            Specifies if all gaussian functions contributing to the main fit must be plotted individually.
        plot_initial_guesses : bool, default=False
            Specifies if the initial guesses should be plotted.
        """
        initial_guesses_array = None
        if plot_initial_guesses:
            initial_guesses = self.get_initial_guesses()
            initial_guesses_array = np.array([
                [peak["mean"], peak["amplitude"]] for peak in initial_guesses.values()
            ])

        base_params = {
            "ax" : ax,
            "fit" : self.fitted_function,
            "initial_guesses" : initial_guesses_array if self.fitted_function else None
        }

        if plot_all and self.fitted_function:
            gaussians = {
                str(i) : models.Gaussian1D(
                    amplitude=self.fit_results.amplitude.value[i], 
                    mean=self.fit_results["mean"].value[i], 
                    stddev=self.fit_results.stddev.value[i]
                ) for i in self.fit_results.index
            }
            self.plot(**base_params, **gaussians)
        else:
            self.plot(**base_params)

    def fit(self) -> models:
        """
        Fits the Spectrum using specutils. This method presupposes the existence of a double peak.

        Returns
        -------
        fitted function : astropy.modeling.core.CompoundModel
            Model of the fitted distribution using two gaussian functions.
        """
        parameter_bounds = {
            "amplitude" : (0, 100)*u.Jy,
            "stddev" : (0, 100)*u.um
        }

        return super().fit(parameter_bounds)

    def get_initial_guesses(self) -> dict:
        """
        Finds the most plausible initial guess for the amplitude and mean value of every gaussian function representing
        a peak in the spectrum.

        Returns
        -------
        initial guesses : dict
            To every ray (key) is associated another dict in which the keys are the amplitude, stddev and mean.
        """
        peaks = find_peaks(
            self.data,
            prominence=self.PEAK_PROMINENCE,
            height=float(np.std(self.data[:100]) * self._PEAK_MINIMUM_HEIGHT_SIGMAS),
            distance=self.PEAK_MINIMUM_DISTANCE
        )[0]

        if list(peaks) != []:
            initial_guesses = {
                i : {"mean" : peak, "amplitude" : self.data[peak], "stddev" : 3} for i, peak in enumerate(peaks)
            }
            return initial_guesses
        else:
            return {}

    def get_FWHM_speed(self, gaussian_function_index: int) -> np.ndarray:
        """
        Gets the full width at half max of a function along with its uncertainty in km/s.

        Parameters
        ----------
        gaussian_function_index : int
            Index of the gaussian function whose FWHM in km/s is desired.

        Returns
        -------
        FWHM : np.ndarray
            FWHM in km/s and its uncertainty measured in km/s.
        """
        channels_FWHM = self.get_FWHM_channels(gaussian_function_index)
        # Get the axis index that represents the velocity by searching for the keyword "VELO-LSR"
        h_axis_velocity = list(self.header.keys())[list(self.header.values()).index("VELO-LSR")][-1]
        return np.abs(channels_FWHM * self.header[f"CDELT{h_axis_velocity}"] / 1000)
    
    @property
    def fit_valid(self) -> bool:
        """
        Checks if the fit is valid by verifying that there is no single gaussian function with a very large stddev.

        Returns
        -------
        bool
            True if the fit is valid, False otherwise.
        """
        MAX_FWHM = 2    # Threshold for maximum accepted FWHM (km/s)
        if self.fit_results is not None:
            if self.fit_results.shape[0] == 1 and self.get_FWHM_speed(0)[0] > MAX_FWHM:
                valid = False
            else:
                valid = True
        else:
            valid = True
        return valid
    
    def get_chi2(self) -> float:
        """
        Gets the chi-square of the fit.

        Returns
        -------
        chi2 : float
            Chi-square of the fit.
        """
        chi2 = np.sum(self.get_subtracted_fit()**2 / np.var(self.data[self._NOISE_CHANNELS]))
        normalized_chi2 = chi2 / self.data.shape[0]
        return normalized_chi2

    def get_FWHM_snr_7_components_array(self) -> np.ndarray:
        raise NotImplementedError
        """
        Get the 7x3 dimensional array representing the FWHM, snr and 7 components fit.
        This method is used in the fits_analyzer.worker_fit() function which creates heavy arrays.

        Returns
        -------
        numpy array: all values in the array are specific to a certain pixel that was fitted. For the first six rows,
        the first element is the FWHM value in km/s, the second element is the uncertainty in km/s and the third
        element is the snr of the peak. The peaks are in the following order: OH1, OH2, OH3, OH4, NII and Ha. The last
        row has only a relevant element in the first column: it takes the value 1 if a double NII peak was considered
        and 0 otherwise. The two other elements are filled with False only to make the array have the same length in
        every dimension.
        """
        return np.array((
            np.concatenate((self.get_FWHM_speed("OH1"), np.array([self.get_snr("OH1")]))),
            np.concatenate((self.get_FWHM_speed("OH2"), np.array([self.get_snr("OH2")]))),
            np.concatenate((self.get_FWHM_speed("OH3"), np.array([self.get_snr("OH3")]))),
            np.concatenate((self.get_FWHM_speed("OH4"), np.array([self.get_snr("OH4")]))),
            np.concatenate((self.get_FWHM_speed("NII"), np.array([self.get_snr("NII")]))),
            np.concatenate((self.get_FWHM_speed("Ha"), np.array([self.get_snr("Ha")]))),
            np.array([self.seven_components_fit, False, False])
        ))

    def get_amplitude_7_components_array(self) -> np.ndarray:
        raise NotImplementedError
        """
        Get the 7x3 dimensional array representing the amplitude of every fitted gaussian function and 7 components
        fit. This method is used in the fits_analyzer.worker_fit() function which creates heavy arrays.

        Returns
        -------
        numpy array: all values in the array are specific to a certain pixel that was fitted. For the first six rows,
        the first element is the amplitude value, the second element is the uncertainty and the third element is False,
        present to make the array have the same shape then the array given by the get_FWHM_snr_7_components_array()
        method. The peaks are in the following order: OH1, OH2, OH3, OH4, NII and Ha. The last row has only a relevant
        element in the first column: it takes the value 1 if a double NII peak was considered and 0 otherwise. The two
        other elements are filled with False only to make the array have the same length in every dimension.
        """
        # If a double NII peak was considered, the mean value between both NII peaks needs to be considered
        if self.seven_components_fit == 0:
            return np.array((
                [self.get_fit_parameters("OH1").amplitude.value, self.get_uncertainties()["OH1"]["amplitude"], False],
                [self.get_fit_parameters("OH2").amplitude.value, self.get_uncertainties()["OH2"]["amplitude"], False],
                [self.get_fit_parameters("OH3").amplitude.value, self.get_uncertainties()["OH3"]["amplitude"], False],
                [self.get_fit_parameters("OH4").amplitude.value, self.get_uncertainties()["OH4"]["amplitude"], False],
                [self.get_fit_parameters("NII").amplitude.value, self.get_uncertainties()["NII"]["amplitude"], False],
                [self.get_fit_parameters("Ha").amplitude.value, self.get_uncertainties()["Ha"]["amplitude"], False],
                [self.seven_components_fit, False, False]
            ))
        else:
            return np.array((
                [self.get_fit_parameters("OH1").amplitude.value, self.get_uncertainties()["OH1"]["amplitude"], False],
                [self.get_fit_parameters("OH2").amplitude.value, self.get_uncertainties()["OH2"]["amplitude"], False],
                [self.get_fit_parameters("OH3").amplitude.value, self.get_uncertainties()["OH3"]["amplitude"], False],
                [self.get_fit_parameters("OH4").amplitude.value, self.get_uncertainties()["OH4"]["amplitude"], False],
                [np.nanmean((self.get_fit_parameters("NII").amplitude.value,
                             self.get_fit_parameters("NII_2").amplitude.value)),
                 np.nanmean((self.get_uncertainties()["NII"]["amplitude"],
                             self.get_uncertainties()["NII_2"]["amplitude"])),
                 False],
                [self.get_fit_parameters("Ha").amplitude.value, self.get_uncertainties()["Ha"]["amplitude"], False],
                [self.seven_components_fit, False, False]
            ))

    def get_mean_7_components_array(self) -> np.ndarray:
        raise NotImplementedError
        """
        Get the 7x3 dimensional array representing the mean of every fitted gaussian function and 7 components fit.
        This method is used in the fits_analyzer.worker_fit() function which creates heavy arrays.

        Returns
        -------
        numpy array: all values in the array are specific to a certain pixel that was fitted. For the first six rows,
        the first element is the mean value, the second element is the uncertainty and the third element is False,
        present to make the array have the same shape then the array given by the get_FWHM_snr_7_components_array()
        method. The peaks are in the following order: OH1, OH2, OH3, OH4, NII and Ha. The last row has only a relevant
        element in the first column: it takes the value 1 if a double NII peak was considered and 0 otherwise. The two
        other elements are filled with False only to make the array have the same length in every dimension.
        """
        # If a double NII peak was considered, the mean value between both NII peaks needs to be considered
        if self.seven_components_fit == 0:
            return np.array((
                [self.get_fit_parameters("OH1").mean.value, self.get_uncertainties()["OH1"]["mean"], False],
                [self.get_fit_parameters("OH2").mean.value, self.get_uncertainties()["OH2"]["mean"], False],
                [self.get_fit_parameters("OH3").mean.value, self.get_uncertainties()["OH3"]["mean"], False],
                [self.get_fit_parameters("OH4").mean.value, self.get_uncertainties()["OH4"]["mean"], False],
                [self.get_fit_parameters("NII").mean.value, self.get_uncertainties()["NII"]["mean"], False],
                [self.get_fit_parameters("Ha").mean.value, self.get_uncertainties()["Ha"]["mean"], False],
                [self.seven_components_fit, False, False]
            ))
        else:
            return np.array((
                [self.get_fit_parameters("OH1").mean.value, self.get_uncertainties()["OH1"]["mean"], False],
                [self.get_fit_parameters("OH2").mean.value, self.get_uncertainties()["OH2"]["mean"], False],
                [self.get_fit_parameters("OH3").mean.value, self.get_uncertainties()["OH3"]["mean"], False],
                [self.get_fit_parameters("OH4").mean.value, self.get_uncertainties()["OH4"]["mean"], False],
                [np.nanmean((self.get_fit_parameters("NII").mean.value,
                             self.get_fit_parameters("NII_2").mean.value)),
                 np.nanmean((self.get_uncertainties()["NII"]["mean"],
                             self.get_uncertainties()["NII_2"]["mean"])),
                 False],
                [self.get_fit_parameters("Ha").mean.value, self.get_uncertainties()["Ha"]["mean"], False],
                [self.seven_components_fit, False, False]
            ))

    def get_list_of_NaN_arrays(self) -> list[np.ndarray]:
        raise NotImplementedError
        """
        Get the 3 elements list of 7x3 arrays filled with NaNs. This is used when a pixel need to be invalidated.

        Returns
        -------
        list: each element in the list is a 7x3 numpy array filled with NaNs.
        """
        return [np.full((7,3), np.NAN), np.full((7,3), np.NAN), np.full((7,3), np.NAN)]

    def is_nicely_fitted_for_NII(self) -> bool:
        raise NotImplementedError
        """
        Check the fit's quality for the NII ray with various conditions.

        Returns
        -------
        bool: True if the fit is usable and False if the fit is poorly made.
        """
        max_residue_limit = 0.2 * self.get_fit_parameters("NII").amplitude.value
        # Check if the maximum residue between channels 10 and 20 is lower than max_residue_limit
        is_max_residue_low = np.max(np.abs(self.get_subtracted_fit()[9:20]))/u.Jy < max_residue_limit
        max_residue_stddev_limit = 0.07 * self.get_fit_parameters("NII").amplitude.value

        # Check if the residue's standard deviation between channels 10 and 20 is lower than max_residue_stddev_limit
        is_residue_stddev_low = self.get_residue_stddev((9,20)) < max_residue_stddev_limit
        return is_max_residue_low and is_residue_stddev_low
