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
            max_residue_sigmas: float=5,
            stddev_detection_threshold: float = 0.5,
            initial_guesses_maximum_gaussian_stddev: float=7,
            initial_guesses_minimum_gaussian_stddev: float=1,
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
        max_residue_sigmas : float, default=6
            Minimum residue signal, in sigmas, at which the fit will not be marked as well fitted. This is used to refit
            abnormal spectrums.
        stddev_detection_threshold : float, default=0.5
            Vertical distance between the half maximum and the spectrum below which the spectrum points will be
            considered for stddev estimation.
        initial_guesses_maximum_gaussian_stddev : float, default=7
            Maximum accepted stddev for the initial guess of each gaussian. This places an upper limit on the stddev the
            algorithm can detect.
        initial_guesses_maximum_gaussian_stddev : float, default=1
            Minimum accepted stddev for the initial guess of each gaussian. This places a lower limit on the stddev the
            algorithm can detect.
        """
        super().__init__(data, header)
        self.PEAK_PROMINENCE = peak_prominence
        self.PEAK_MINIMUM_HEIGHT_SIGMAS = peak_minimum_height_sigmas
        self.PEAK_MINIMUM_DISTANCE = peak_minimum_distance
        self.PEAK_WIDTH = peak_width
        self.NOISE_CHANNELS = noise_channels
        self.INITIAL_GUESSES_BINNING = initial_guesses_binning
        self.MAX_RESIDUE_SIGMAS = max_residue_sigmas
        self.STDDEV_DETECTION_THRESHOLD = stddev_detection_threshold
        self.RESIDUE_RATIO_AMPLITUDE_MODIFIER = 1.1 # defines the amount by which the maximum residue peak will be
        # shifted when given as an initial guess when refitting, which accounts for the fitting algorithm's broadness
        self.INITIAL_GUESSES_MAXIMUM_GAUSSIAN_STDDEV = initial_guesses_maximum_gaussian_stddev
        self.INITIAL_GUESSES_MINIMUM_GAUSSIAN_STDDEV = initial_guesses_minimum_gaussian_stddev

    @property
    def y_threshold(self):
        return float(np.std(self.data[self.NOISE_CHANNELS]) * self.PEAK_MINIMUM_HEIGHT_SIGMAS)

    def fit(self, parameter_bounds: dict={}):
        """
        Fits the Spectrum using specutils. This method presupposes the existence of a double peak.

        Parameters
        ----------
        parameter_bounds : dict
            Bounds of every parameter for every gaussian. 
            Example : {"amplitude": (0, 8)*u.Jy, "stddev": (0, 1)*u.um, "mean": (20, 30)*u.um}.
        """
        default_parameter_bounds = {
            "amplitude" : (0, 100)*u.Jy,
            "stddev" : (1e-5, 100)*u.um,     # Prevent division by zero
            "mean" : (0, len(self))*u.um    
        }

        super().fit(default_parameter_bounds | parameter_bounds)

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

        # Correct for a binning factor
        for i in range(len(peaks[0])):
            peaks[0][i] = peaks[0][i]*self.INITIAL_GUESSES_BINNING

        if list(peaks[0]) != []:
            # Triggers if the fit is done a second time
            # This is used to enhance the fit's quality and detect unfitted components
            if self.initial_guesses:
                mean = np.argmax(np.abs(self.get_subtracted_fit()))
                self.initial_guesses[len(self.initial_guesses)] = {
                    "mean" : mean + 1,
                    "amplitude" : np.max(np.abs(self.get_subtracted_fit())) * self.RESIDUE_RATIO_AMPLITUDE_MODIFIER,
                    "stddev" : 7.2      # value chosen from the mean of multiple successful fits
                }

            max_ig, min_ig = self.INITIAL_GUESSES_MAXIMUM_GAUSSIAN_STDDEV, self.INITIAL_GUESSES_MINIMUM_GAUSSIAN_STDDEV
            for i in range(len(peaks[0])):
                # The FWHM is estimated to be given as a stddev estimate
                # The estimated intersection at half maximum is calculated
                intersects = np.abs(self.data - peaks[1]["peak_heights"][i]/2) < self.STDDEV_DETECTION_THRESHOLD
                lower_half_width = np.argmax(np.flip(intersects[:peaks[0][i] + 1], axis=0))
                upper_half_width = np.argmax(intersects[peaks[0][i]:])
                stddev = min(lower_half_width, upper_half_width) / (np.sqrt(2*np.log(2)))

                # + 1 accounts for the fact that scipy uses 0-based indexing and headers/ds9 use 1-based indexing
                self.initial_guesses[i] = {
                    "mean" : peaks[0][i] + 1,
                    "amplitude" : peaks[1]["peak_heights"][i],
                    "stddev" : min(max_ig, max(min_ig, stddev))
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
