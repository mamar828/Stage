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
            peak_minimum_height_sigmas: float=5.0,
            peak_minimum_distance: int=10,
            peak_width: int=3,
            noise_channels: slice=slice(0,100),
            initial_guesses_binning: int=1,
            max_residue_sigmas: int=6
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
        peak_width : int, default=3
            Minimum width acceptable to detect a peak. This is used in the scipy.signal.find_peak function.
        noise_channels : slice, default=slice(0,100)
            Channels used to measure the noise's stddev. No peaks should be found in this region. 
        initial_guesses_binning : int, default=1
            Factor by which to bin the data to find the initial guesses. If kept at 1, the initial guesses are found
            with the raw data.
        max_residue_sigmas : int, default=6
            Minimum residue signal, in sigmas, at which the fit will not be marked as well fitted. This is used to refit
            abnormal spectrums.
        """
        super().__init__(data, header)
        self.PEAK_PROMINENCE = peak_prominence
        self.PEAK_MINIMUM_HEIGHT_SIGMAS = peak_minimum_height_sigmas
        self.PEAK_MINIMUM_DISTANCE = peak_minimum_distance
        self.PEAK_WIDTH = peak_width
        self.NOISE_CHANNELS = noise_channels
        self.INITIAL_GUESSES_BINNING = initial_guesses_binning
        self.MAX_RESIDUE_SIGMAS = max_residue_sigmas

    @property
    def y_threshold(self):
        noise_channels = slice(self.NOISE_CHANNELS.start // self.INITIAL_GUESSES_BINNING,
                               self.NOISE_CHANNELS.stop // self.INITIAL_GUESSES_BINNING)
        return float(np.std(self.data[noise_channels]) * self.PEAK_MINIMUM_HEIGHT_SIGMAS)

    @Spectrum.fit_needed
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
            initial_guesses_array = np.array([
                [peak["mean"], peak["amplitude"]] for peak in self.initial_guesses.values()
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
            "stddev" : (1e-5, 100)*u.um
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
        if self.INITIAL_GUESSES_BINNING > 1:
            s = self.bin(self.INITIAL_GUESSES_BINNING)
            # s.auto_plot()
            data = s.data
        else:
            data = self.data

        peaks = find_peaks(
            data,
            prominence=self.PEAK_PROMINENCE,
            height=self.y_threshold,
            distance=self.PEAK_MINIMUM_DISTANCE / self.INITIAL_GUESSES_BINNING,
            width=self.PEAK_WIDTH / self.INITIAL_GUESSES_BINNING
        )

        if list(peaks[0]) != []:
            # Triggers if the fit is done a second time
            # This is used to enhance the fit's quality
            if self.initial_guesses:
                mean = np.argmax(np.abs(self.get_subtracted_fit()))
                self.initial_guesses[len(self.initial_guesses)] = {
                    "mean" : mean + 1,
                    "amplitude" : self.data[mean],
                    "stddev" : 3
                }

            # + 1 accounts for the fact that scipy uses 0-based indexing and headers/ds9 use 1-based indexing
            for i in range(len(peaks[0])):
                self.initial_guesses[i] = {
                    "mean" : peaks[0][i]*self.INITIAL_GUESSES_BINNING + 1,
                    "amplitude" : peaks[1]["peak_heights"][i],
                    "stddev" : 3
                }
            
            return self.initial_guesses
        else:
            return {}

    @Spectrum.fit_needed
    def get_FWHM_speed(self, gaussian_function_index: int) -> np.ndarray:
        """
        Gives the full width at half max of a function along with its uncertainty in km/s.

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
    @Spectrum.fit_needed
    def is_well_fitted(self) -> bool:
        """
        Checks if the fit is well done by verifying that there is no large peak in the fit's residue.

        Returns
        -------
        good fit : bool
            True if the Spectrum is well fitted, False otherwise.
        """
        good_fit = np.max(np.abs(self.get_subtracted_fit())) \
                 < self.get_residue_stddev(self.NOISE_CHANNELS) * self.MAX_RESIDUE_SIGMAS
        return good_fit
