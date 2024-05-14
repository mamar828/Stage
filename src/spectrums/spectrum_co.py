from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import scipy
from astropy.modeling import models, fitting
from astropy.io import fits
from astropy import units as u
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines
from eztcolors import Colors as C

from src.headers.header import Header
from src.spectrums.spectrum import Spectrum


class SpectrumCO(Spectrum):
    """
    Encapsulates the methods specific to CO spectrums.
    """

    def plot_fit(self, text: str=None, fullscreen: bool=False, plot_all: bool=False, plot_initial_guesses: bool=False):
        """
        Sends the fitted functions and the subtracted fit to the plot() method.

        Parameters
        ----------
        text : str, default=None.
            Text to be displayed as the title of the plot. This is used for debugging purposes.
        fullscreen : bool, default=False
            Specifies if the graph must be opened in fullscreen.
        plot_all : bool, default=False
            Specifies if all gaussian functions contributing to the main fit must be plotted individually.
        plot_initial_guesses : bool, default=False
            Specifies if the initial guesses should be plotted.
        """
        if plot_initial_guesses:
            i = self.get_initial_guesses()
            initial_guesses_array = np.array([
                [i["OH1"]["x0"], i["OH2"]["x0"], i["OH3"]["x0"], i["OH4"]["x0"], i["NII"]["x0"], i["Ha"]["x0"]],
                [i["OH1"]["a"],  i["OH2"]["a"],  i["OH3"]["a"],  i["OH4"]["a"],  i["NII"]["a"],  i["Ha"]["a"]]
            ])
        else:
            initial_guesses_array = None

        base_params = {
            "title" : text,
            "fullscreen" : fullscreen,
            "fit" : self.fit,
            "subtracted_fit" : self.get_subtracted_fit(),
            "initial_guesses" : initial_guesses_array
        }

        if plot_all:
            g = self.fit
            # Define the functions to be plotted
            oh1 = models.Gaussian1D(amplitude=g.amplitude_0.value, mean=g.mean_0.value, stddev=g.stddev_0.value)
            oh2 = models.Gaussian1D(amplitude=g.amplitude_1.value, mean=g.mean_1.value, stddev=g.stddev_1.value)
            oh3 = models.Gaussian1D(amplitude=g.amplitude_2.value, mean=g.mean_2.value, stddev=g.stddev_2.value)
            oh4 = models.Gaussian1D(amplitude=g.amplitude_3.value, mean=g.mean_3.value, stddev=g.stddev_3.value)
            nii = models.Gaussian1D(amplitude=g.amplitude_4.value, mean=g.mean_4.value, stddev=g.stddev_4.value)
            ha  = models.Gaussian1D(amplitude=g.amplitude_5.value, mean=g.mean_5.value, stddev=g.stddev_5.value)
            self.plot(**base_params, OH1=oh1, OH2=oh2, OH3=oh3, OH4=oh4, NII=nii, Ha=ha)

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
            0 : {"amplitude": (0, 8)*u.Jy},
            1 : {"amplitude": (0, 8)*u.Jy}
        } # stddev and mean also possible (*u.um)

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
        guesses = {}
        # Trial and error determined value that allows the best detection of a peak by measuring the difference between
        # consecutive derivatives
        diff_threshold = -0.45
        # Trial and error determined value that acts specifically on the determination of the OH3 peak by looking at
        # the difference between consecutive derivatives in the case that no peak is present
        diff_threshold_OH3 = 1.8

        derivatives = np.zeros(shape=(47,2))
        for i in range(0, len(self.x_values)-1):
            derivatives[i,0] = i + 1
            derivatives[i,1] = self.y_values[i+1] - self.y_values[i]

        # Create a list to compute the difference between consecutive derivatives
        derivatives_diff = []
        for x in range(2,48):
            # Simplify the indices in lists (channel one is element zero of a list)
            x_list = x - 1
            derivatives_diff.append(derivatives[x_list,1] - derivatives[x_list-1,1])
        # Note that the first element of the list is the derivative difference between channels 2 and 3 and channels 1
        # and 2

        x_peaks = {}
        for ray, bounds in [("OH1", (1,5)), ("OH2", (19,22)), ("OH3", (36,39)), ("OH4", (47,48)),
                            ("NII", (13,17)), ("Ha", (42,45))]:
            # Initial x value of the peak
            x_peak = 0
            # Separate value for the OH3 ray that predominates on x_peak if a distinct peak is found
            x_peak_OH3 = 0
            # Specify when a big difference in derivatives has been detected and allows to keep the x_value
            stop_OH3 = False
            # Consecutive drops signify a very probable peak in the vicinity
            consecutive_drops_OH2 = 0
            # For the OH1 and OH4 rays, the maximum intensity is used as the initial guess
            if ray != "OH1" and ray != "OH4":
                for x in range(bounds[0], bounds[1]):
                    # Variables used to ease the use of lists
                    current_derivatives_diff = derivatives_diff[x-2]
                    current_y_value = self.y_values[x-1]
                    if ray == "OH2":
                        if current_derivatives_diff < 0.5:     # A minor rise is also considered for consecutive 
                                                               # "drops"
                            consecutive_drops_OH2 += 1
                        else:
                            consecutive_drops_OH2 = 0
                        if consecutive_drops_OH2 == 2:
                            # 2 consecutive drops are interpreted as a ray
                            x_peaks[ray] = x - 1
                            break

                    if ray == "OH3":
                        # First condition checks if a significant change in derivative is noticed which could indicate
                        # a peak
                        # Also makes sure that the peak is higher than any peak that might have been found previously
                        if current_derivatives_diff < diff_threshold and (
                            current_y_value > self.y_values[x_peak_OH3-1] or x_peak_OH3 == 0):
                            x_peak_OH3 = x

                        # In the case that no peak is noticed, the second condition checks when the derivative
                        # suddenly rises
                        # This can indicate a "bump" in the emission ray's shape, betraying the presence of another
                        # component
                        # This condition is only True once as the derivatives keep rising after the "bump"
                        if current_derivatives_diff > diff_threshold_OH3 and not stop_OH3:
                            x_peak = x
                            stop_OH3 = True

                    else:
                        # For other rays, only a significant change in derivative is checked while making sure it is
                        # the max value
                        if current_derivatives_diff < diff_threshold and (
                            current_y_value > self.y_values[x_peak-1] or x_peak == 0):
                            x_peak = x
            if consecutive_drops_OH2 != 2:
                # If no peak is found, the peak is chosen to be the maximum value within the bounds
                if x_peak == 0:
                    x_peak = bounds[0] + np.argmax(self.y_values[bounds[0]-1:bounds[1]-1])

                # If a real peak has been found for the OH3 ray, it predominates over a big rise in derivative
                if x_peak_OH3 != 0:
                    x_peak = x_peak_OH3

                x_peaks[ray] = x_peak

        for ray in ["OH1", "OH2", "OH3", "OH4", "NII", "Ha"]:
            guesses[ray] = {"x0": x_peaks[ray], "a": self.y_values[x_peaks[ray]-1]}
        return guesses

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
        speed_FWHM = scipy.constants.c * angstroms_FWHM[0] / angstroms_center[0] / 1000
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
