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
        self.PEAK_MINIMUM_HEIGHT_SIGMAS = peak_minimum_height_sigmas
        self.PEAK_MINIMUM_DISTANCE = peak_minimum_distance
        self.NOISE_CHANNELS = noise_channels
        self.y_threshold = np.std(self.data[self.NOISE_CHANNELS]) * self.PEAK_MINIMUM_HEIGHT_SIGMAS

    def plot_fit(self, ax: Axes, plot_all: bool=False, plot_initial_guesses: bool=False):
        """
        Sends the fitted functions to the plot() method to be plotted on an axis.

        Parameters
        ----------
        ax : Axes
            Axis on which to plot the Array2D.
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
            if isinstance(self.fitted_function, models.Gaussian1D):
                gaussians = {
                    "0" : models.Gaussian1D(
                        amplitude=self.fitted_function.amplitude.value, 
                        mean=self.fitted_function.mean.value, 
                        stddev=self.fitted_function.stddev.value
                    )
                }
            else:
                gaussians = {
                    str(i) : models.Gaussian1D(
                        amplitude=self.fitted_function[i].amplitude.value, 
                        mean=self.fitted_function[i].mean.value, 
                        stddev=self.fitted_function[i].stddev.value
                    ) for i in range(len(self.fitted_function.parameters)//3)
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
            height=float(np.std(self.data[:100]) * self.PEAK_MINIMUM_HEIGHT_SIGMAS),
            distance=self.PEAK_MINIMUM_DISTANCE
        )[0]

        if list(peaks) != []:
            initial_guesses = {
                i : {"mean" : peak, "amplitude" : self.data[peak], "stddev" : 5} for i, peak in enumerate(peaks)
            }
            return initial_guesses
        else:
            return {}

    def get_uncertainties(self) -> dict:
        """
        Gets the uncertainty on every parameter of every gaussian component.

        Returns
        -------
        uncertainties : dict
            Every key is the index of a gaussian function and the value is another dict with the uncertainty values
            linked to the keys "amplitude", "mean" and "stddev".
        """
        cov_matrix = self.fitted_function.meta["fit_info"]["param_cov"]
        uncertainty_matrix = np.sqrt(np.diag(cov_matrix))
        # The uncertainty matrix is stored as a_0, x0_0, sigma_0, a_1, x0_1, sigma_1, ...
        ordered_uncertainties = {
            i : {
                "amplitude": uncertainty_matrix[3*i],
                "mean": uncertainty_matrix[3*i+1],
                "stddev": uncertainty_matrix[3*i+2]
            } for i in range(len(uncertainty_matrix)//3)
        }
        return ordered_uncertainties

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
        spectral_length = self.header["FP_I_A"]
        wavelength_channel_1 = self.header["FP_B_L"]
        number_of_channels = self.header["NAXIS3"]
        params = getattr(self.fitted_function, gaussian_function_index)
        uncertainties = self.get_uncertainties()[gaussian_function_index]
        channels_FWHM = self.get_FWHM_channels(gaussian_function_index)

        angstroms_center = np.array((params.mean.value, uncertainties["mean"])) * spectral_length / number_of_channels
        angstroms_center[0] +=  wavelength_channel_1
        angstroms_FWHM = channels_FWHM * spectral_length / number_of_channels
        speed_FWHM = c * angstroms_FWHM[0] / angstroms_center[0] / 1000
        speed_FWHM_uncertainty = speed_FWHM * (angstroms_FWHM[1]/angstroms_FWHM[0] +
                                               angstroms_center[1]/angstroms_center[0])
        speed_array = np.array((speed_FWHM, speed_FWHM_uncertainty))
        return speed_array

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
