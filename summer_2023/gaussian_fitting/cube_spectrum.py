from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy

from astropy.modeling import models, fitting
from astropy.io import fits
from astropy import units as u

from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines



class Spectrum():
    """
    Encapsulate all the methods of any data cube's spectrum.
    """

    def __init__(self, data: np.ndarray, header: fits.Header):
        """
        Initialize a Spectrum object with a certain header, whose spectral information will be taken.

        Arguments
        ---------
        data: numpy array. Detected intensity at each channel.
        header: astropy.io.fits.Header. Allows for the calculation of the FWHM using the interferometer's settings.
        """
        self.x_values, self.y_values = np.arange(len(data)) + 1, data
        self.header = header

    @classmethod
    def from_dat_file(cls, filename: str, header: fits.Header, cube_number: int=None) -> Spectrum:
        """
        Create a Spectrum object from a .dat file. This is used for testing specific spectrums.
        
        Arguments
        ---------
        filename: str. Name of the .dat file.
        header: fits.Header. Header of the data cube from which the spectrum was extracted.
        cube_number: int, default=None. In the case of an SII spectrum, specifies the number of the data cube from 
        which the spectrum was extracted. This is useful because not all data cubes present the peaks at the same 
        channels.
        """
        lines = open(filename, "r").readlines()
        y_values = []
        for point in lines[:-1]:
            y_values.append(float(point.split(" ")[-1]))

        if cube_number is None:
            return cls(np.array(y_values), header)
        else:
            # It is a spectrum from one of the SII cubes
            return cls(np.array(y_values), header, cube_number)

    def plot(self, coords: tuple[int]=None, fullscreen: bool=False, **other_values):
        """
        Plot the spectrum along with the fit's residue, if a fit was made. If plotting the fit is desired, the 
        plot_fit() method should be used as it wraps this method in case of plotting fits.

        Arguments
        ---------
        coords: tuple of ints, default=None. x and y coordinates of the evaluated point that serve as a landmark in the
        cube and will appear on screen. This is used for debugging purposes.
        fullscreen: bool, default=False. Specifies if the graph must be opened in fullscreen.
        other_values. This argument may take any distribution to be plotted and is used to plot all the gaussian fits
        on the same plot. The name used for each keyword argument will be present in the plot's legend.
        """
        fig, axs = plt.subplots(2)
        axs[0].plot(self.x_values, self.y_values, "g-", label="ds9 spectrum", linewidth=3, alpha=0.6)
        for name, value in other_values.items():
            if value is None:
                continue
            x_plot_gaussian = np.arange(1,self.x_values[-1]+0.05,0.05)
            # The following conditions only allow to use a specific color when plotting certain gaussian functions or
            # other distributions or to choose a particular subplot
            if name == "fit":
                # Fitted entire function
                axs[0].plot(x_plot_gaussian*u.Jy, value(x_plot_gaussian*u.um), "r-", label=name)
            elif name == "subtracted_fit":
                # Residual distribution
                axs[1].plot(self.x_values, value, label=name)
            elif name == "NII":
                # NII gaussian
                axs[0].plot(x_plot_gaussian, value(x_plot_gaussian), "m-", label=name, linewidth="1")
            elif name == "NII_2":
                # Second NII gaussian
                axs[0].plot(x_plot_gaussian, value(x_plot_gaussian), "c-", label=name, linewidth="1")
            elif name == "SII1":
                # First SII gaussian
                axs[0].plot(x_plot_gaussian, value(x_plot_gaussian), "m-", label=name, linewidth="1")
            elif name == "SII2":
                # Second SII gaussian
                axs[0].plot(x_plot_gaussian, value(x_plot_gaussian), "c-", label=name, linewidth="1")
            elif name == "initial_guesses":
                # Simple points plotting
                axs[0].plot(value[0], value[1], "kx", label=name, markersize="10")
            else:
                # Fitted individual gaussians
                axs[0].plot(x_plot_gaussian, value(x_plot_gaussian), "y-", label=name, linewidth="1")
        
        axs[0].legend(loc="upper left", fontsize="7")
        axs[1].legend(loc="upper left", fontsize="7")
        plt.xlabel("channels")
        axs[0].set_ylabel("intensity")
        axs[1].set_ylabel("intensity")

        if coords:
            fig.text(0.4, 0.89, f"coords: {coords}")

        if fullscreen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        plt.show()

    def get_residue_stddev(self, bounds: tuple[int]=None) -> float:
        """
        Get the standard deviation of the fit's residue.

        Arguments
        ---------
        bounds: tuple[int], default=None. Bounds between which the residue's standard deviation should be calculated.
        If None is provided, the residue's stddev is calculated for all x. Bounds indexing is the same as lists,
        e.g. bounds=(0,2) gives x=1 and x=2.

        Returns
        -------
        float: value of the residue's standard deviation.
        """
        if bounds is None:
            return np.std(self.get_subtracted_fit()/u.Jy)
        else:
            return np.std(self.get_subtracted_fit()[slice(*bounds)]/u.Jy)

    def get_subtracted_fit(self) -> np.ndarray:
        """
        Get the subtracted fit's values.

        Returns
        -------
        numpy array: result values of the gaussian fit subtracted to the y values.
        """
        subtracted_y = self.y_values*u.Jy - self.fitted_gaussian(self.x_values*u.um)
        return subtracted_y

    def get_FWHM_channels(self, peak_name: str) -> np.ndarray:
        """
        Get the full width at half maximum of a gaussian function along with its uncertainty in channels.

        Arguments
        ---------
        peak_name: str. Name of the peak whose FWHM in channels is desired. The supported peaks are the one present
        in the get_fit_parameters() and get_uncertainties() methods.

        Returns
        -------
        numpy array: array of the FWHM and its uncertainty measured in channels.
        """
        fwhm = 2*np.sqrt(2*np.log(2))*self.get_fit_parameters(peak_name).stddev.value
        fwhm_uncertainty = 2*np.sqrt(2*np.log(2))*self.get_uncertainties()[peak_name]["stddev"]
        return np.array((fwhm, fwhm_uncertainty))

    def get_snr(self, peak_name: str) -> float:
        """
        Get the signal to noise ratio of a peak. This is calculated as the amplitude of the peak divided by the
        residue's standard deviation.

        Arguments
        ---------
        peak_name: str. Name of the peak whose amplitude will be used to calculate the snr.

        Returns
        -------
        float: value of the signal to noise ratio.
        """
        return (self.get_fit_parameters(peak_name).amplitude/u.Jy)/self.get_residue_stddev()



class Calibration_spectrum(Spectrum):
    """
    Encapsulate the methods specific to calibration spectrums.
    """

    def __init__(self, data: np.ndarray, header: fits.Header):
        """
        Initialize a Calibration_spectrum object. The fitter will use a single gaussian.

        Arguments
        ---------
        data: numpy array. Detected intensity at each channel.
        header: astropy.io.fits.Header. Allows for the calculation of the FWHM using the interferometer's settings.
        """
        super().__init__(data, header)

        # Application of a translation
        desired_peak_position = 35
        upper_limit_mean_calculation = 25

        # The distance between the desired peak and the current peak is calculated
        peak_position_translation = desired_peak_position - (list(self.y_values).index(max(self.y_values)) + 1)
        # The values are translated to keep the peak always at the same channel
        self.y_values = np.roll(self.y_values, peak_position_translation)
        # All y values are shifted downwards by the mean calculated in the 25 first channels
        mean = np.sum(self.y_values[0:upper_limit_mean_calculation]) / upper_limit_mean_calculation
        self.y_values -= mean
        # A tuple containing the peak's x and y is stored
        self.max_tuple = (int(self.x_values[desired_peak_position - 1]),
                          float(self.y_values[desired_peak_position - 1]))

    def plot_fit(self, coords: tuple=None, fullscreen: bool=False):
        """
        Send the fitted function and the subtracted fit to the plot() method.

        Arguments
        ---------
        coords: tuple of ints, default=None. x and y coordinates of the evaluated point that serve as a landmark in the
        cube and will appear on screen. This is used for debugging purposes.
        fullscreen: bool, default=False. Specifies if the graph must be opened in fullscreen.
        """
        self.plot(coords, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit())

    def fit(self) -> models:
        """
        Fit the cube's data using specutils. Also set the astropy model of the fitted gaussian to the variable
        self.fitted_gaussian.

        Returns
        -------
        astropy.modeling.core.CompoundModel: model of the fitted distribution using a single gaussian function.
        """
        spectrum_object = Spectrum1D(flux=self.y_values*u.Jy, spectral_axis=self.x_values*u.um)
        # Initialize the single gaussian using the max peak's position
        g_init = models.Gaussian1D(amplitude=self.max_tuple[1]*u.Jy, mean=self.max_tuple[0]*u.um)
        try:
            self.fitted_gaussian = fit_lines(spectrum_object, g_init,
                                fitter=fitting.LMLSQFitter(calc_uncertainties=True), get_fit_info=True, maxiter=1000)
            return self.fitted_gaussian
        except:
            # Sometimes the fit is unsuccessful with uncommon spectrums at the borders of the calibration cube
            pass

    def get_fit_parameters(self, peak_name: str=None) -> models:
        """
        Get the parameters of the fit's gaussian function.

        Arguments
        ---------
        peak_name: str, default=None. Specifies from which peak the function needs to be extracted. The output is
        independent of the provided argument and the latter is only kept to increase class methods compatibility.

        Returns
        -------
        astropy.modeling.core.CompoundModel: function representing the fitted calibration's peak
        """
        return self.fitted_gaussian

    def get_uncertainties(self) -> dict:
        """
        Get the uncertainty on every parameter of the gaussian function.

        Returns
        -------
        dict: the key is "calibration" and the value is another dict with the uncertainty values linked to the keys
        "amplitude", "mean" and "stddev". The "calibration" key is only kept to increase class methods compatibility.
        """
        cov_matrix = self.fitted_gaussian.meta["fit_info"]["param_cov"]
        uncertainty_matrix = np.sqrt(np.diag(cov_matrix))
        # The values in the uncertainty matrix are stored as a_0, x0_0, sigma_0, a_1, x0_1, sigma_1, ...
        return {"calibration": {"amplitude": uncertainty_matrix[0], "mean": uncertainty_matrix[1],
                                "stddev": uncertainty_matrix[2]}}

    def get_FWHM_speed(self, peak_name: str=None) -> np.ndarray:
        """
        Get the full width at half maximum of the gaussian function along with its uncertainty in km/s.

        Arguments
        ---------
        peak_name: str, default=None. Specifies which peak's FWHM in km/s is desired. The output is independent of the
        provided argument and the latter is only kept to increase class methods compatibility.

        Returns
        -------
        numpy array: array of the FWHM and its uncertainty measured in km/s.
        """
        # Extract the necessary values from the cube's header
        spectral_length = self.header["FP_I_A"]
        wavelength_channel_1 = self.header["FP_B_L"]
        number_of_channels = self.header["NAXIS3"]
        params = self.get_fit_parameters()
        uncertainties = self.get_uncertainties()["calibration"]
        channels_FWHM = self.get_FWHM_channels("calibration")

        angstroms_center = np.array((params.mean.value, uncertainties["mean"])) * spectral_length / number_of_channels
        angstroms_center[0] += wavelength_channel_1
        angstroms_FWHM = channels_FWHM * spectral_length / number_of_channels
        speed_FWHM = scipy.constants.c * angstroms_FWHM[0] / angstroms_center[0] / 1000
        speed_FWHM_uncertainty = speed_FWHM * (angstroms_FWHM[1]/angstroms_FWHM[0] +
                                               angstroms_center[1]/angstroms_center[0])
        return np.array((speed_FWHM, speed_FWHM_uncertainty))



class NII_spectrum(Spectrum):
    """
    Encapsulate the methods specific to NII spectrums.
    """

    def __init__(self, data: np.ndarray, header: fits.Header, seven_components_fit_authorized: bool=False):
        """
        Initialize a NII_spectrum object. The fitter will use multiple gaussians.

        Arguments
        ---------
        data: numpy array. Detected intensity at each channel.
        header: astropy.io.fits.Header. Allows for the calculation of the FWHM using the interferometer's settings.
        seven_components_fit_authorized: bool, default=False. Specifies if a fit with seven components, i.e. two NII
        components, can be detected and used.
        """
        super().__init__(data, header)
        # The seven_components_fit variable takes the value 1 if a seven component fit was done in the NII cube
        self.seven_components_fit = 0
        self.seven_components_fit_authorized = seven_components_fit_authorized

        # All y values are shifted downwards by the mean calculated in the channels 25 to 35
        self.downwards_shift = np.sum(self.y_values[24:34]) / 10
        self.y_values -= self.downwards_shift

    def plot_fit(self, coords: tuple[int]=None, fullscreen: bool=False, plot_all: bool=False,
                 plot_initial_guesses: bool=False):
        """
        Send the fitted functions and the subtracted fit to the plot() method.

        Arguments
        ---------
        coords: tuple of ints, default=None. x and y coordinates of the evaluated point that serve as a landmark in the
        cube and will appear on screen. This is used for debugging purposes.
        fullscreen: bool, default=False. Specifies if the graph must be opened in fullscreen.
        plot_all: bool, default=False. Specifies if all gaussian functions contributing to the main fit must be plotted
        individually.
        plot_initial_guesses: bool, default=False. Specifies if the initial guesses should be plotted.
        """
        if plot_initial_guesses:
            i = self.get_initial_guesses()
            initial_guesses_array = np.array([
                [i["OH1"]["x0"], i["OH2"]["x0"], i["OH3"]["x0"], i["OH4"]["x0"], i["NII"]["x0"], i["Ha"]["x0"]],
                [i["OH1"]["a"],  i["OH2"]["a"],  i["OH3"]["a"],  i["OH4"]["a"],  i["NII"]["a"],  i["Ha"]["a"]]
            ])
        else:
            initial_guesses_array = None

        if plot_all:
            g = self.fitted_gaussian
            # Define the functions to be plotted
            oh1 = models.Gaussian1D(amplitude=g.amplitude_0.value, mean=g.mean_0.value, stddev=g.stddev_0.value)
            oh2 = models.Gaussian1D(amplitude=g.amplitude_1.value, mean=g.mean_1.value, stddev=g.stddev_1.value)
            oh3 = models.Gaussian1D(amplitude=g.amplitude_2.value, mean=g.mean_2.value, stddev=g.stddev_2.value)
            oh4 = models.Gaussian1D(amplitude=g.amplitude_3.value, mean=g.mean_3.value, stddev=g.stddev_3.value)
            nii = models.Gaussian1D(amplitude=g.amplitude_4.value, mean=g.mean_4.value, stddev=g.stddev_4.value)
            ha  = models.Gaussian1D(amplitude=g.amplitude_5.value, mean=g.mean_5.value, stddev=g.stddev_5.value)
            try:
                nii_2 = models.Gaussian1D(amplitude=g.amplitude_6.value, mean=g.mean_6.value, stddev=g.stddev_6.value)
                self.plot(coords, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit(),
                          OH1=oh1, OH2=oh2, OH3=oh3, OH4=oh4, NII=nii, Ha=ha, NII_2=nii_2,
                          initial_guesses=initial_guesses_array)
            except:
                self.plot(coords, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit(),
                        OH1=oh1, OH2=oh2, OH3=oh3, OH4=oh4, NII=nii, Ha=ha, initial_guesses=initial_guesses_array)

        else:
            self.plot(coords, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit(),
                      initial_guesses=initial_guesses_array)

    def fit(self, number_of_components: int=6) -> models:
        """
        Fit the data cube using specutils and initial guesses. Also sets the astropy model of the fitted gaussian to
        the variable self.fitted_gaussian. This method supposes that four polluting peaks are present in the 
        spectrum and they will be called OHX.

        Arguments
        ---------
        number_of_components: int, default=6. Number of initial guesses that need to be returned. This integer may be
        6 or 7 depending on if a double NII peak is detected. The user may leave the default value as it is as the
        program will attempt a seven components fit if allowed and if needed.

        Returns
        -------
        astropy.modeling.core.CompoundModel: model of the fitted distribution using 6 or 7 gaussian functions.
        """
        # Initialize the six gaussians using the guesses dict
        guesses = self.get_initial_guesses(number_of_components)
        # The parameter bounds dictionary allows for greater accuracy and limits each parameters with values found
        # by trial and error
        parameter_bounds = {
            "OH1": {"amplitude": (0, 100)*u.Jy,
                    "stddev": (np.sqrt(guesses["OH1"]["a"])/5, np.sqrt(guesses["OH1"]["a"])/2)*u.um},
            "OH2": {"amplitude": (0, 15-self.downwards_shift)*u.Jy,
                    "stddev": (np.sqrt(guesses["OH2"]["a"])/5, np.sqrt(guesses["OH2"]["a"])/2)*u.um,
                    "mean": (19,21)*u.um},
            "OH3": {"amplitude": (0, 13-self.downwards_shift)*u.Jy,
                    "stddev": (np.sqrt(guesses["OH3"]["a"])/5, np.sqrt(guesses["OH3"]["a"])/2)*u.um,
                    "mean": (36,39)*u.um},
            "OH4": {"amplitude": (0, 100)*u.Jy,
                    "stddev": (np.sqrt(guesses["OH4"]["a"])/5, np.sqrt(guesses["OH4"]["a"])/2)*u.um}
        }
        if number_of_components == 6:
            parameter_bounds["NII"] = {"amplitude": (0,100)*u.Jy, "mean": (12,17)*u.um}
            parameter_bounds["Ha"]  = {"amplitude": (0,100)*u.Jy, "mean": (41,45)*u.um}
        else:
            amplitude_mean = np.mean((guesses["NII"]["a"], guesses["NII_2"]["a"]))
            parameter_bounds["NII"]   = {
                "amplitude": (amplitude_mean/1.6, amplitude_mean*1.6)*u.Jy,
                "stddev": (np.sqrt(guesses["NII"]["a"])/6, np.sqrt(guesses["NII"]["a"])/2)*u.um,
                "mean": (12,14.5)*u.um
            }
            parameter_bounds["NII_2"] = {
                "amplitude": (amplitude_mean/1.6, amplitude_mean*1.6)*u.Jy,
                "stddev": (np.sqrt(guesses["NII_2"]["a"])/6, np.sqrt(guesses["NII_2"]["a"])/2)*u.um,
                "mean": (14.5,17)*u.um
            }
            parameter_bounds["Ha"]    = {
                "amplitude": (0,100)*u.Jy,
                "stddev": (np.sqrt(guesses["Ha"]["a"])/10, np.sqrt(guesses["Ha"]["a"])/1.6),
                "mean": (41,45)*u.um
            }

        spectrum = Spectrum1D(flux=self.y_values*u.Jy, spectral_axis=self.x_values*u.um)
        gi_OH1 = models.Gaussian1D(amplitude=guesses["OH1"]["a"]*u.Jy, mean=guesses["OH1"]["x0"]*u.um,
                                   bounds=parameter_bounds["OH1"])
        gi_OH2 = models.Gaussian1D(amplitude=guesses["OH2"]["a"]*u.Jy, mean=guesses["OH2"]["x0"]*u.um,
                                   bounds=parameter_bounds["OH2"])
        gi_OH3 = models.Gaussian1D(amplitude=guesses["OH3"]["a"]*u.Jy, mean=guesses["OH3"]["x0"]*u.um,
                                   bounds=parameter_bounds["OH3"])
        gi_OH4 = models.Gaussian1D(amplitude=guesses["OH4"]["a"]*u.Jy, mean=guesses["OH4"]["x0"]*u.um,
                                   bounds=parameter_bounds["OH4"])
        gi_NII = models.Gaussian1D(amplitude=guesses["NII"]["a"]*u.Jy, mean=guesses["NII"]["x0"]*u.um,
                                   bounds=parameter_bounds["NII"])
        gi_Ha  = models.Gaussian1D(amplitude=guesses["Ha"]["a"] *u.Jy, mean=guesses["Ha"]["x0"] *u.um,
                                   bounds=parameter_bounds["Ha"])
        gi_OH1.mean.max = 3*u.um
        gi_OH4.mean.min = 47*u.um
        if number_of_components == 6:
            self.fitted_gaussian = fit_lines(spectrum, gi_OH1 + gi_OH2 + gi_OH3 + gi_OH4 + gi_NII + gi_Ha,
                            fitter=fitting.LMLSQFitter(calc_uncertainties=True), get_fit_info=True, maxiter=10000)
            if not self.seven_components_fit_authorized:
                return self.fitted_gaussian
            else:
                # Check the possibility of a double-component NII peak
                nii_FWHM = self.get_FWHM_speed("NII")[0]
                ha_FWHM  = self.get_FWHM_speed("Ha")[0]
                if nii_FWHM > ha_FWHM or nii_FWHM > 40:
                    self.seven_components_fit = 1
                    return self.fit_NII_cube(number_of_components=7)
                else:
                    return self.fitted_gaussian

        else:   # Seven components fit
            gi_NII_2 = models.Gaussian1D(amplitude=guesses["NII_2"]["a"]*u.Jy, mean=guesses["NII_2"]["x0"]*u.um,
                                         bounds=parameter_bounds["NII_2"])
            self.fitted_gaussian = fit_lines(spectrum, gi_OH1 + gi_OH2 + gi_OH3 + gi_OH4 + gi_NII + gi_Ha + gi_NII_2,
                            fitter=fitting.LMLSQFitter(calc_uncertainties=True), get_fit_info=True, maxiter=10000)
            return self.fitted_gaussian

    def get_initial_guesses(self, number_of_components: int=6) -> dict:
        """
        Find the most plausible initial guess for the amplitude and mean value of every gaussian function representing
        a peak in the spectrum.

        Arguments
        ---------
        number_of_components: int, default=6. Number of initial guesses that need to be returned. This integer may be 6
        or 7 depending on if a double NII peak is visible.

        Returns
        -------
        dict: to every ray (key) is associated another dict in which the keys are the amplitude "a" and the mean value
        "x0".
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
            if ray == "NII" and number_of_components == 7:
                # If a double peak was detected, the initial guesses are hard-coded to obtain better results
                x_peaks["NII"], x_peaks["NII_2"] = 13, 16
                continue
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
        if number_of_components == 7:
            guesses["NII_2"] = {"x0": x_peaks["NII_2"], "a": self.y_values[x_peaks["NII_2"]-1]}
        return guesses

    def get_fit_parameters(self, peak_name: str=None) -> models:
        """
        Get the parameters of a gaussian component of the complete fit.

        Arguments
        ---------
        peak_name: str, default=None. Specifies from which peak the function needs to be extracted. The supported peaks
        are: "OH1", "OH2", "OH3", "OH4", "NII", "Ha" and "NII_2", the latter only a seven components fit was made.

        Returns
        -------
        astropy.modeling.core.CompoundModel: function representing the specified peak
        """
        # The rays are stored in the CompoundModel in the same order than the following dict
        peak_numbers = {"OH1": 0, "OH2": 1, "OH3": 2, "OH4": 3, "NII": 4, "Ha": 5, "NII_2": 6}
        return self.fitted_gaussian[peak_numbers[peak_name]]

    def get_uncertainties(self) -> dict:
        """
        Get the uncertainty on every parameter of every gaussian component.

        Returns
        -------
        dict: every key is a function ("OH1", "OH2", ...) and the value is another dict with the uncertainty values
        linked to the keys "amplitude", "mean" and "stddev".
        """
        cov_matrix = self.fitted_gaussian.meta["fit_info"]["param_cov"]
        uncertainty_matrix = np.sqrt(np.diag(cov_matrix))
        # The uncertainty matrix is stored as a_0, x0_0, sigma_0, a_1, x0_1, sigma_1, ...
        ordered_uncertainties = {}
        for i, peak_name in zip(range(int(len(uncertainty_matrix)/3)), (["OH1", "OH2", "OH3", "OH4", "NII", "Ha"])):
            ordered_uncertainties[peak_name] = {
                "amplitude": uncertainty_matrix[3*i],
                "mean": uncertainty_matrix[3*i+1],
                "stddev": uncertainty_matrix[3*i+2]
            }
        # Check if the fit was done with seven components
        if len(uncertainty_matrix)/3 == 7:
            ordered_uncertainties["NII_2"] = {
                "amplitude": uncertainty_matrix[3*6],
                "mean": uncertainty_matrix[3*6+1],
                "stddev": uncertainty_matrix[3*6+2]
            }
        return ordered_uncertainties

    def get_FWHM_speed(self, peak_name: str=None) -> np.ndarray:
        """
        Get the full width at half max of a function along with its uncertainty in km/s.

        Arguments
        ---------
        peak_name: str, default=None. Name of the peak whose FWHM in km/s is desired. The supported peaks are:
        "OH1", "OH2", "OH3", "OH4", "NII" and "Ha". If a two-components NII fit was made, the FWHM value is the mean
        value of both NII peaks.

        Returns
        -------
        numpy array: array of the FWHM and its uncertainty measured in km/s.
        """
        spectral_length = self.header["FP_I_A"]
        wavelength_channel_1 = self.header["FP_B_L"]
        number_of_channels = self.header["NAXIS3"]
        params = self.get_fit_parameters(peak_name)
        uncertainties = self.get_uncertainties()[peak_name]
        channels_FWHM = self.get_FWHM_channels(peak_name)

        angstroms_center = np.array((params.mean.value, uncertainties["mean"])) * spectral_length / number_of_channels
        angstroms_center[0] +=  wavelength_channel_1
        angstroms_FWHM = channels_FWHM * spectral_length / number_of_channels
        speed_FWHM = scipy.constants.c * angstroms_FWHM[0] / angstroms_center[0] / 1000
        speed_FWHM_uncertainty = speed_FWHM * (angstroms_FWHM[1]/angstroms_FWHM[0] +
                                               angstroms_center[1]/angstroms_center[0])
        speed_array = np.array((speed_FWHM, speed_FWHM_uncertainty))
        # Check if the NII peak is used and if a double fit was done
        if peak_name == "NII":
            try:
                return np.mean((speed_array, self.get_FWHM_speed("NII_2")), axis=0)
            except:
                pass
        return speed_array

    def get_FWHM_snr_7_components_array(self) -> np.ndarray:
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
        """
        Get the 3 elements list of 7x3 arrays filled with NaNs. This is used when a pixel need to be invalidated.

        Returns
        -------
        list: each element in the list is a 7x3 numpy array filled with NaNs.
        """
        return [np.full((7,3), np.nan), np.full((7,3), np.nan), np.full((7,3), np.nan)]

    def is_nicely_fitted_for_NII(self) -> bool:
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



class SII_spectrum(Spectrum):
    """
    Encapsulate the methods specific to SII spectrums.
    """

    def __init__(self, data: np.ndarray, header: fits.Header, cube_number: int):
        """
        Initialize a SII_spectrum object. The fitter will use multiple gaussians.

        Arguments
        ---------
        data: numpy array. Detected intensity at each channel.
        header: astropy.io.fits.Header. Allows for the calculation of the FWHM using the interferometer's settings.
        cube_number: int. Number of the SII_data_cube from which the spectrum comes. This is used because all SII cubes
        do not present the rays at the same places.
        """
        super().__init__(data, header)
        # The seven_components_fit variable takes the value 1 if a seven component fit was done in the NII cube

        # The bounds are set depending on the cube's type
        # The OH1 and OH2 rays refer to the two polluting rays
        if cube_number == 1:
            self.mean_bounds = {"OH1": (13,16), "OH2": (41,44), "SII1": (7,12), "SII2": (35,40)}
            self.downwards_shift = np.mean(self.y_values[19:29])
        elif cube_number == 2:
            self.mean_bounds = {"OH1": (16,20), "OH2": (45,48), "SII1": (11,14), "SII2": (39,43)}
            self.downwards_shift = np.mean(self.y_values[22:33])
        elif cube_number == 3:
            self.mean_bounds = {"OH1": (14,17), "OH2": (42,45), "SII1": (8,11), "SII2": (38,41)}
            self.downwards_shift = np.mean(self.y_values[19:29])
        else:
            raise ValueError(f"Invalid cube_number, must be between 1 and 3. Provided cube_number: {cube_number}")
        # All y values are shifted downwards by the mean calculated between certain channels
        self.y_values -= self.downwards_shift

    def plot_fit(self, coords: tuple[int]=None, fullscreen: bool=False, plot_all: bool=False,
                 plot_initial_guesses: bool=False):
        """
        Send the fitted functions and the subtracted fit to the plot() method.

        Arguments
        ---------
        coords: tuple of ints, default=None. x and y coordinates of the evaluated point that serve as a landmark in the
        cube and will appear on screen. This is used for debugging purposes.
        fullscreen: bool, default=False. Specifies if the graph must be opened in fullscreen.
        plot_all: bool, default=False. Specifies if all gaussian functions contributing to the main fit must be plotted
        individually.
        plot_initial_guesses: bool, default=False. Specifies if the initial guesses should be plotted.
        """
        if plot_initial_guesses:
            i = self.get_initial_guesses()
            initial_guesses_array = np.array([
                [i["OH1"]["x0"], i["OH2"]["x0"], i["SII1"]["x0"], i["SII2"]["x0"]],
                [i["OH1"]["a"],  i["OH2"]["a"],  i["SII1"]["a"],  i["SII2"]["a"]]
            ])
        else:
            initial_guesses_array = None

        if plot_all:
            g = self.fitted_gaussian
            # Define the functions to be plotted
            oh1  = models.Gaussian1D(amplitude=g.amplitude_0.value, mean=g.mean_0.value, stddev=g.stddev_0.value)
            oh2  = models.Gaussian1D(amplitude=g.amplitude_1.value, mean=g.mean_1.value, stddev=g.stddev_1.value)
            sii1 = models.Gaussian1D(amplitude=g.amplitude_2.value, mean=g.mean_2.value, stddev=g.stddev_2.value)
            sii2 = models.Gaussian1D(amplitude=g.amplitude_3.value, mean=g.mean_3.value, stddev=g.stddev_3.value)
            self.plot(coords, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit(),
                      OH1=oh1, OH2=oh2, SII1=sii1, SII2=sii2, initial_guesses=initial_guesses_array)

        else:
            self.plot(coords, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit(),
                      initial_guesses=initial_guesses_array)

    def fit(self) -> models:
        """
        Fit the data cube using specutils and initial guesses. Also sets the astropy model of the fitted gaussian to
        the variable self.fitted_gaussian. This method supposes that two polluting rays are present in the spectrum 
        and will be refered to as OH1 and OH2.

        Returns
        -------
        astropy.modeling.core.CompoundModel: model of the fitted distribution using 4 gaussian functions.
        """
        # Initialize the six gaussians using the guesses dict
        guesses = self.get_initial_guesses()
        # The parameter bounds dictionary allows for greater accuracy and limits each parameters with values found
        # by trial and error
        parameter_bounds = {
            "OH1" : {"amplitude": (0,10-self.downwards_shift)*u.Jy, "mean": self.mean_bounds["OH1"]*u.um,
                     "stddev": (np.sqrt(guesses["OH1"]["a"])/2, np.sqrt(guesses["OH1"]["a"])/1)*u.um},
            "OH2" : {"amplitude": (0,9-self.downwards_shift)*u.Jy, "mean": self.mean_bounds["OH2"]*u.um,
                     "stddev": (np.sqrt(guesses["OH2"]["a"])/2, np.sqrt(guesses["OH2"]["a"])/1)*u.um},
            "SII1": {"amplitude": (0,100)*u.Jy, "mean": self.mean_bounds["SII1"]*u.um},
            "SII2": {"amplitude": (0,100)*u.Jy, "mean": self.mean_bounds["SII2"]*u.um}
        }

        spectrum = Spectrum1D(flux=self.y_values*u.Jy, spectral_axis=self.x_values*u.um)
        gi_OH1  = models.Gaussian1D(amplitude=guesses["OH1"]["a"]*u.Jy, mean=guesses["OH1"]["x0"]*u.um,
                                    bounds=parameter_bounds["OH1"])
        gi_OH2  = models.Gaussian1D(amplitude=guesses["OH2"]["a"]*u.Jy, mean=guesses["OH2"]["x0"]*u.um,
                                    bounds=parameter_bounds["OH2"])
        gi_SII1 = models.Gaussian1D(amplitude=guesses["SII1"]["a"]*u.Jy, mean=guesses["SII1"]["x0"]*u.um,
                                    bounds=parameter_bounds["SII1"])
        gi_SII2 = models.Gaussian1D(amplitude=guesses["SII2"]["a"]*u.Jy, mean=guesses["SII2"]["x0"]*u.um,
                                    bounds=parameter_bounds["SII2"])
        gi_SII1.stddev.max = np.sqrt(guesses["SII2"]["a"])
        gi_SII2.stddev.max = np.sqrt(guesses["SII2"]["a"])

        self.fitted_gaussian = fit_lines(spectrum, gi_OH1 + gi_OH2 + gi_SII1 + gi_SII2,
                            fitter=fitting.LMLSQFitter(calc_uncertainties=True), get_fit_info=True, maxiter=10000)
        return self.fitted_gaussian

    def get_initial_guesses(self) -> dict:
        """
        Find the most plausible initial guess for the amplitude and mean value of every gaussian function representing 
        a peak in the spectrum.

        Returns
        -------
        dict: to every ray (key) is associated another dict in which the keys are the amplitude "a" and the mean value
        "x0".
        """
        guesses = {}
        # Trial and error determined value that allows the best detection of a peak by measuring the difference between
        # consecutive derivatives
        diff_threshold = 0.45

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

        x_peaks = {}
        for ray, bounds, needed_drops in [("OH1", self.mean_bounds["OH1"], 3), ("OH2", self.mean_bounds["OH2"], 2),
                                        ("SII1", self.mean_bounds["SII1"], 4), ("SII2", self.mean_bounds["SII2"], 4)]:
            # The needed_drops variable sets the number of consecutive drops that is needed to attribute this link to 
            # a peak
            # Initial x value of the peak depending on the derivative diff
            x_peak = {"up": 0, "down": 0}
            # Initialize a variable to examine the global ray's shape
            consecutive_drops = 0
            for x in range(bounds[0], bounds[1]+1):
                if x == 48:
                    break
                # Create variables used to ease the use of lists
                current_derivatives_diff = derivatives_diff[x - 2]
                x_list = x - 1
                if current_derivatives_diff < 0.1:     # A minor rise is also considered for consecutive "drops"
                    consecutive_drops += 1
                else:
                    consecutive_drops = 0
                if consecutive_drops == needed_drops:
                    # The number specified by needed_drops of consecutive drops are interpreted as a ray
                    x_peaks[ray] = x - 1
                    break
                if current_derivatives_diff > diff_threshold and (
                    self.y_values[x_list] > self.y_values[x_peak["up"]-1] or x_peak["up"] == 0):
                    # Significant change in derivatives + maximum value that has this significant bump for a positive
                    # change
                    x_peak["up"] = x
                elif current_derivatives_diff < -diff_threshold and (
                    self.y_values[x_list] > self.y_values[x_peak["down"]-1] or x_peak["down"] == 0):
                    # Significant change in derivatives + maximum value that has this significant bump for a negative
                    # change
                    x_peak["down"] = x

            if consecutive_drops != needed_drops:
                # If no peak is found, the peak is chosen to be the maximum value within the bounds
                if x_peak == {"up": 0, "down": 0}:
                    x_peaks[ray] = bounds[0] + np.argmax(self.y_values[bounds[0]-1:bounds[1]-1])
                else:
                    # Peaks that are detected with negative derivatives differences are prioritized
                    if x_peak["down"] != 0:
                        x_peaks[ray] = x_peak["down"]
                    else:
                        x_peaks[ray] = x_peak["up"]

        for ray in ["OH1", "OH2", "SII1", "SII2"]:
            guesses[ray] = {"x0": x_peaks[ray], "a": self.y_values[x_peaks[ray]-1]}
        return guesses

    def get_fit_parameters(self, peak_name: str=None) -> models:
        """
        Get the parameters of a gaussian component of the complete fit.

        Arguments
        ---------
        peak_name: str, default=None. Specifies from which peak the function needs to be extracted. The supported peaks
        are: "OH1", "OH2", "SII1" and "SII2".

        Returns
        -------
        astropy.modeling.core.CompoundModel: function representing the specified peak.
        """
        # The rays are stored in the CompoundModel in the same order than the following dict
        peak_numbers = {"OH1": 0, "OH2": 1, "SII1": 2, "SII2": 3}
        return self.fitted_gaussian[peak_numbers[peak_name]]

    def get_uncertainties(self) -> dict:
        """
        Get the uncertainty on every parameter of every gaussian component.

        Returns
        -------
        dict: every key is a function ("OH1", "OH2", ...) and the value is another dict with the uncertainty values
        linked to the keys "amplitude", "mean" and "stddev".
        """
        cov_matrix = self.fitted_gaussian.meta["fit_info"]["param_cov"]
        uncertainty_matrix = np.sqrt(np.diag(cov_matrix))
        # The uncertainty matrix is stored as a_0, x0_0, sigma_0, a_1, x0_1, sigma_1, ...
        ordered_uncertainties = {}
        for i, peak_name in zip(range(int(len(uncertainty_matrix)/3)), (["OH1", "OH2", "SII1", "SII2"])):
            ordered_uncertainties[peak_name] = {
                "amplitude": uncertainty_matrix[3*i], 
                "mean": uncertainty_matrix[3*i+1], 
                "stddev": uncertainty_matrix[3*i+2]
            }
        return ordered_uncertainties

    def get_FWHM_speed(self, peak_name: str=None) -> np.ndarray:
        """
        Get the full width at half max of a function along with its uncertainty in km/s.

        Arguments
        ---------
        peak_name: str default=None. Name of the peak whose FWHM in km/s is desired. The supported peaks are:
        "OH1", "OH2", "SII1" and "SII2".

        Returns
        -------
        numpy array: array of the FWHM and its uncertainty measured in km/s.
        """
        spectral_length = self.header["FP_I_A"]
        wavelength_channel_1 = self.header["FP_B_L"]
        number_of_channels = self.header["NAXIS3"]
        params = self.get_fit_parameters(peak_name)
        uncertainties = self.get_uncertainties()[peak_name]
        channels_FWHM = self.get_FWHM_channels(peak_name)

        angstroms_center = np.array((params.mean.value, uncertainties["mean"])) * spectral_length / number_of_channels
        angstroms_center[0] += wavelength_channel_1
        angstroms_FWHM = channels_FWHM * spectral_length / number_of_channels
        speed_FWHM = scipy.constants.c * angstroms_FWHM[0] / angstroms_center[0] / 1000
        speed_FWHM_uncertainty = speed_FWHM * (angstroms_FWHM[1]/angstroms_FWHM[0] +
                                               angstroms_center[1]/angstroms_center[0])
        speed_array = np.array((speed_FWHM, speed_FWHM_uncertainty))
        return speed_array

    def get_FWHM_snr_array(self) -> np.ndarray:
        """
        Get the 4x3 dimensional array representing the FWHM and snr of each element in the Spectrum.
        This method is used in the fits_analyzer.worker_fit() function which creates heavy arrays.

        Returns
        -------
        numpy array: all values in the array are specific to a certain pixel that was fitted. For all four rows, the
        first element is the FWHM value in km/s, the second element is the uncertainty in km/s and the third element is 
        the snr of the peak. The peaks are in the following order: OH1, OH2, SII1 and SII2.
        """
        return np.array((
            np.concatenate((self.get_FWHM_speed("OH1"), np.array([self.get_snr("OH1")]))),
            np.concatenate((self.get_FWHM_speed("OH2"), np.array([self.get_snr("OH2")]))),
            np.concatenate((self.get_FWHM_speed("SII1"), np.array([self.get_snr("SII1")]))),
            np.concatenate((self.get_FWHM_speed("SII2"), np.array([self.get_snr("SII2")])))
        ))

    def get_amplitude_array(self) -> np.ndarray:
        """
        Get the 4x3 dimensional array representing the amplitude of each fitted gaussian function.
        This method is used in the fits_analyzer.worker_fit() function which creates heavy arrays.

        Returns
        -------
        numpy array: all values in the array are specific to a certain pixel that was fitted. For all four rows, the 
        first element is the amplitude value, the second element is the uncertainty and the third element is False, 
        present to make the array have the same shape then the array given by the get_FWHM_snr_array() method. The
        peaks are in the following order: OH1, OH2, SII1 and SII2.
        """
        return np.array((
            [self.get_fit_parameters("OH1").amplitude.value, self.get_uncertainties()["OH1"]["amplitude"], False],
            [self.get_fit_parameters("OH2").amplitude.value, self.get_uncertainties()["OH2"]["amplitude"], False],
            [self.get_fit_parameters("SII1").amplitude.value, self.get_uncertainties()["SII1"]["amplitude"], False],
            [self.get_fit_parameters("SII2").amplitude.value, self.get_uncertainties()["SII2"]["amplitude"], False]
        ))

    def get_mean_array(self) -> np.ndarray:
        """
        Get the 4x3 dimensional array representing the mean of every fitted gaussian function.
        This method is used in the fits_analyzer.worker_fit() function which creates heavy arrays.

        Returns
        -------
        numpy array: all values in the array are specific to a certain pixel that was fitted. For all four rows, the
        first element is the mean value, the second element is the uncertainty and the third element is False, present
        to make the array have the same shape then the array given by the get_FWHM_snr_array() method. The peaks are in
        the following order: OH1, OH2, SII1 and SII2.
        """
        return np.array((
            [self.get_fit_parameters("OH1").mean.value, self.get_uncertainties()["OH1"]["mean"], False],
            [self.get_fit_parameters("OH2").mean.value, self.get_uncertainties()["OH2"]["mean"], False],
            [self.get_fit_parameters("SII1").mean.value, self.get_uncertainties()["SII1"]["mean"], False],
            [self.get_fit_parameters("SII2").mean.value, self.get_uncertainties()["SII2"]["mean"], False]
        ))

    def get_list_of_NaN_arrays(self) -> list[np.ndarray]:
        """
        Get the 3 elements list of 4x3 arrays filled with NaNs. This is used when a pixel needs to be invalidated.

        Returns
        -------
        list: each element in the list is a 4x3 numpy array filled with NaNs.
        """
        return [np.full((4,3), np.nan), np.full((4,3), np.nan), np.full((4,3), np.nan)]

    def is_nicely_fitted(self) -> bool:
        """
        Check the fit's quality with various conditions.

        Returns
        -------
        bool: True if the fit is usable and False if the fit is poorly made.
        """
        max_residue_limit = 1.15
        # Check if the maximum residue between channels 3 and 20 and between channels 30 and 44 is lower than 
        # max_residue_limit
        is_max_residue_low = np.max(np.abs(np.concatenate([self.get_subtracted_fit()[2:20],
                                                          self.get_subtracted_fit()[29:44]])))/u.Jy < max_residue_limit
        max_residue_stddev_limit = 0.55
        # Check if the total residue's standard deviation is lower than max_residue_stddev_limit
        is_residue_stddev_low = self.get_residue_stddev() < max_residue_stddev_limit
        return is_max_residue_low and is_residue_stddev_low



# Test area
""" 
def loop_di_loop(filename):
    x = 150
    # calib: 490, 493
    iter_n = open("gaussian_fitting/other/iter_number.txt", "r").read()
    for y in range(int(iter_n), 1013):
        print(f"\n----------------\ncoords: {x,y}")
        data = fits.open(filename)[0].data
        header = fits.open(filename)[0].header
        spectrum = SII_spectrum(data[:,y-1,x-1], header, 1)
        # spectrum = Calibration_spectrum(data[:,y-1,x-1], header)
        spectrum.fit()
        print(spectrum.fitted_gaussian)
        # print(spectrum.get_FWHM_speed("calibration"))
        print(spectrum.get_FWHM_speed("SII1"))
        print(spectrum.get_FWHM_speed("SII2"))
        # spectrum.plot_fit(fullscreen=True, coords=(x,y))
        # if spectrum.get_residue_stddev() < 0.55:
        print(spectrum.get_residue_stddev())
            # print((spectrum.get_FWHM_speed("SII1")-spectrum.get_FWHM_speed("SII2"))/spectrum.get_FWHM_speed("SII2"))
        spectrum.plot_fit(fullscreen=True, coords=(x,y), plot_initial_guesses=True, plot_all=True)
        file = open("gaussian_fitting/other/iter_number.txt", "w")
        file.write(str(y+1))
        file.close()
# loop_di_loop("ref1.fits")
# loop_di_loop("gaussian_fitting/data_cubes/SII/SII_2/calibration.fits")

def loop_di_loop(filename):
    x = 300
    # calib: 490, 493
    iter_n = open("gaussian_fitting/other/iter_number.txt", "r").read()
    for y in range(int(iter_n), 1013):
        print(f"\n----------------\ncoords: {x,y}")
        data = fits.open(filename)[0].data
        header = fits.open(filename)[0].header
        spectrum = NII_spectrum(data[:,y-1,x-1], header)
        # spectrum = Calibration_spectrum(data[:,y-1,x-1], header)
        spectrum.fit()
        print(spectrum.fitted_gaussian)
        # print(spectrum.get_FWHM_speed("calibration"))
        print(spectrum.get_FWHM_speed("NII"))
        # spectrum.plot_fit(fullscreen=True, coords=(x,y))
        # if spectrum.get_residue_stddev() < 0.55:
        print(spectrum.get_residue_stddev())
        print(spectrum.is_nicely_fitted_for_NII())
            # print((spectrum.get_FWHM_speed("SII1")-spectrum.get_FWHM_speed("SII2"))/spectrum.get_FWHM_speed("SII2"))
        spectrum.plot_fit(fullscreen=True, coords=(x,y), plot_initial_guesses=True, plot_all=True)
        file = open("gaussian_fitting/other/iter_number.txt", "w")
        file.write(str(y+1))
        file.close()

# loop_di_loop("gaussian_fitting/data_cubes/night_34_binned.fits")
# loop_di_loop("gaussian_fitting/data_cubes/SII/SII_2/calibration.fits")
 """