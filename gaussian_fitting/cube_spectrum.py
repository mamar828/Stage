
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os

from astropy.modeling import models, fitting
from astropy.io import fits
from astropy import units as u

from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines


class Spectrum:
    """
    Encapsulate all the data and methods of a cube's spectrum.
    """

    def __init__(self, data: np.ndarray, header, calibration: bool, seven_components_fit_authorized: bool=False):
        """
        Initialize a Spectrum object. Calibration boolean must be set to True to force the analysis of a single peak such as
        with a calibration cube's spectrum.
        
        Arguments
        ---------
        data: numpy array. Flux at each channel.
        header: astropy.io.fits.header.Header. Allows for the calculation of the FWHM using the spectrometer's settings.
        calibration: bool. Specifies if the fit is for the calibration cube i.e. to fit a single peak. If False, the fitter will
        attempt a 6 components fit.
        seven_components_fit_authorized: bool, default=False. Specifies if a fit with seven components, i.e. two NII components,
        can be detected and used.
        """
        self.x_values, self.y_values = np.arange(len(data)) + 1, data
        self.calibration = calibration
        self.data = data
        # The seven_components_fit variable takes the value 1 if a seven component fit was done in the NII cube
        self.seven_components_fit = 0
        self.header = header
        self.seven_components_fit_authorized = seven_components_fit_authorized

        if calibration:
            # Application of a translation in the case of the calibration cube
            if len(self.x_values) == 48:
                desired_peak_position = 35
                upper_limit_mean_calculation = 25
            if len(self.x_values) == 34:
                desired_peak_position = 20
                upper_limit_mean_calculation = 10
            # The distance between the desired peak and the current peak is calculated
            peak_position_translation = desired_peak_position - (list(self.y_values).index(max(self.y_values)) + 1)
            self.y_values = np.roll(self.y_values, peak_position_translation)
            # All y values are shifted downwards by the mean calculated in the 25 first channels
            mean = np.sum(self.y_values[0:upper_limit_mean_calculation]) / upper_limit_mean_calculation
            self.y_values -= mean
            # A tuple containing the peak's x and y is stored
            self.max_tuple = (int(self.x_values[desired_peak_position - 1]), float(self.y_values[desired_peak_position - 1]))

        else:
            # All y values are shifted downwards by the mean calculated in the channels 25 to 35
            self.downwards_shift = np.sum(self.y_values[24:34]) / 10
            self.y_values -= self.downwards_shift
        
    def plot(self, coords: tuple=None, fullscreen: bool=False, **other_values):
        """
        Plot the data and the fits.
        
        Arguments
        ---------
        coords: tuple of ints, optional. x and y coordinates of the evaluated point. Serves as a landmark in the cube
        and will appear on screen.
        fullscreen: bool, default=False. Specifies if the graph must be opened in fullscreen.
        other_values: optional. This argument may take any distribution to be plotted and is used to plot all the gaussian
        fits on the same plot.
        """
        fig, axs = plt.subplots(2)
        # Plot of the data
        axs[0].plot(self.x_values, self.y_values, "g-", label="ds9 spectrum", linewidth=3, alpha=0.6)
        for name, value in other_values.items():
            x_plot_gaussian = np.arange(1,self.x_values[-1]+0.05,0.05)
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

    def plot_fit(self, coords: int=None, fullscreen: bool=False, plot_all: bool=False):
        """
        Send all the functions to be plotted to the plot method depending on the data cube used.

        Arguments
        ---------
        coord: tuple of ints, optional. x and y coordinates of the evaluated point. Serves as a landmark in the cube
        and will appear on screen.
        fullscreen: bool, default=False. Specifies if the graph must be opened in fullscreen.
        plot_all: bool, default=False. Specifies if all gaussian functions contributing to the main fit must be plotted.
        """
        if plot_all and not self.calibration:
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
                          OH1=oh1, OH2=oh2, OH3=oh3, OH4=oh4, NII=nii, Ha=ha, NII_2=nii_2)
            except:
                self.plot(coords, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit(),
                        OH1=oh1, OH2=oh2, OH3=oh3, OH4=oh4, NII=nii, Ha=ha)
        
        else:
            self.plot(coords, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit())

    def fit_calibration(self) -> models:
        """
        Fit the calibration cube's data using specutils. Also sets the astropy model of the fitted gaussian to the variable
        self.fitted_gaussian.

        Returns
        -------
        astropy.modeling.core.CompoundModel: model of the fitted distribution using a single gaussian function.
        """
        spectrum = Spectrum1D(flux=self.y_values*u.Jy, spectral_axis=self.x_values*u.um)
        # Initialize the single gaussian using the max peak's position
        g_init = models.Gaussian1D(amplitude=self.max_tuple[1]*u.Jy, mean=self.max_tuple[0]*u.um)
        try:
            self.fitted_gaussian = fit_lines(spectrum, g_init,
                                            fitter=fitting.LMLSQFitter(calc_uncertainties=True), get_fit_info=True, maxiter=1000)  
            return self.fitted_gaussian
        except:
            # Fit unsuccessful
            pass

    def fit_NII_cube(self, stddev_mins: dict=None, number_of_components: int=6) -> models:
        """
        Fit the data cube using specutils and initial guesses. Also sets the astropy model of the fitted gaussian to the
        variable self.fitted_gaussian.

        Arguments
        ---------
        stddev_mins: dict, optional. Specifies the standard deviation's minimum value of every gaussian component.
        This is used in the fit_iteratively method to increase the fit's accuracy.
        number_of_components: int, default=6. Number of initial guesses that need to be returned. This integer may be 6 or 7
        depending on if a double NII peak is detected. The user may leave the default value as it is as the program will
        attempt a seven components fit if needed.

        Returns
        -------
        astropy.modeling.core.CompoundModel: model of the fitted distribution using 6 gaussian functions.
        """
        # Initialize the six gaussians using the params dict
        params = self.get_initial_guesses(number_of_components)
        # The parameter bounds dictionary allows for greater accuracy and limits each parameters with values found 
        # by trial and error
        parameter_bounds = {
            "OH1": {"amplitude": (0, 100)*u.Jy,
                    "stddev": (np.sqrt(params["OH1"]["a"])/5, np.sqrt(params["OH1"]["a"])/2)*u.um},
            "OH2": {"amplitude": (0, 15-self.downwards_shift)*u.Jy,
                    "stddev": (np.sqrt(params["OH2"]["a"])/5, np.sqrt(params["OH2"]["a"])/2)*u.um,
                    "mean": (19,21)*u.um},
            "OH3": {"amplitude": (0, 13-self.downwards_shift)*u.Jy,
                    "stddev": (np.sqrt(params["OH3"]["a"])/5, np.sqrt(params["OH3"]["a"])/2)*u.um,
                    "mean": (36,39)*u.um},
            "OH4": {"amplitude": (0, 100)*u.Jy,
                    "stddev": (np.sqrt(params["OH4"]["a"])/5, np.sqrt(params["OH4"]["a"])/2)*u.um}
        }
        if number_of_components == 6:
            parameter_bounds["NII"] = {"amplitude": (0,100)*u.Jy, "mean": (12,17)*u.um}
            parameter_bounds["Ha"]  = {"amplitude": (0,100)*u.Jy, "mean": (41,45)*u.um}
        else:
            amplitude_mean = np.mean((params["NII"]["a"], params["NII_2"]["a"]))
            parameter_bounds["NII"]   = {"amplitude": (amplitude_mean/1.6, amplitude_mean*1.6)*u.Jy,
                                         "stddev": (np.sqrt(params["NII"]["a"])/6, np.sqrt(params["NII"]["a"])/2)*u.um,
                                         "mean": (12,14.5)*u.um}
            parameter_bounds["NII_2"] = {"amplitude": (amplitude_mean/1.6, amplitude_mean*1.6)*u.Jy,
                                         "stddev": (np.sqrt(params["NII_2"]["a"])/6, np.sqrt(params["NII_2"]["a"])/2)*u.um,
                                         "mean": (14.5,17)*u.um}
            parameter_bounds["Ha"]    = {"amplitude": (0,100)*u.Jy,
                                         "stddev": (np.sqrt(params["Ha"]["a"])/10, np.sqrt(params["Ha"]["a"])/1.6),
                                         "mean": (41,45)*u.um}
        
        spectrum = Spectrum1D(flux=self.y_values*u.Jy, spectral_axis=self.x_values*u.um)
        gi_OH1 = models.Gaussian1D(amplitude=params["OH1"]["a"]*u.Jy, mean=params["OH1"]["x0"]*u.um, 
                                   bounds=parameter_bounds["OH1"])
        gi_OH2 = models.Gaussian1D(amplitude=params["OH2"]["a"]*u.Jy, mean=params["OH2"]["x0"]*u.um,
                                   bounds=parameter_bounds["OH2"])
        gi_OH3 = models.Gaussian1D(amplitude=params["OH3"]["a"]*u.Jy, mean=params["OH3"]["x0"]*u.um, 
                                   bounds=parameter_bounds["OH3"])
        gi_OH4 = models.Gaussian1D(amplitude=params["OH4"]["a"]*u.Jy, mean=params["OH4"]["x0"]*u.um, 
                                   bounds=parameter_bounds["OH4"])
        gi_NII = models.Gaussian1D(amplitude=params["NII"]["a"]*u.Jy, mean=params["NII"]["x0"]*u.um,
                                   bounds=parameter_bounds["NII"])
        gi_Ha  = models.Gaussian1D(amplitude=params["Ha"]["a"] *u.Jy, mean=params["Ha"]["x0"] *u.um,
                                   bounds=parameter_bounds["Ha"])
        gi_OH1.mean.max = 3*u.um
        gi_OH4.mean.min = 47*u.um

        # Set the standard deviation's minimum of the gaussians of the corresponding rays if the dict is present
        if stddev_mins:
            for ray, min_guess in stddev_mins.items():
                exec(f"gi_{ray}.stddev.min = {min_guess}*u.um")
        
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
            gi_NII_2 = models.Gaussian1D(amplitude=params["NII_2"]["a"]*u.Jy, mean=params["NII_2"]["x0"]*u.um,
                                         bounds=parameter_bounds["NII_2"])
            self.fitted_gaussian = fit_lines(spectrum, gi_OH1 + gi_OH2 + gi_OH3 + gi_OH4 + gi_NII + gi_Ha + gi_NII_2,
                                             fitter=fitting.LMLSQFitter(calc_uncertainties=True), get_fit_info=True, maxiter=10000)
            return self.fitted_gaussian

    def fit_iteratively(self, stddev_increments: float=0.2) -> models:
        """
        Use the fit method iteratively to find the best possible standard deviation values for the gauss functions representing
        the OH emission rays by minimizing the residual's standard deviation. After finding every best minimum standard
        deviation value, a fit is made with those values.

        Arguments
        ---------
        stddev_increments: float. Indicates the increments that will be used to test every standard deviation value for
        every ray. A smaller value may provide better results, but will also take more time.

        Returns
        -------
        astropy.modeling.core.CompoundModel: model of the fitted distribution using 6 gauss functions. Also sets the astropy
        model to the variable self.fitted_gaussian.
        """
        stddev_mins = {}
        initial_guesses = self.get_initial_guesses()
        for ray in ["OH1", "OH2", "OH3", "OH4"]:
            # Value of the first standard deviation guess
            # Is not set to zero to avoid division by zero
            min_guess = 10 ** (-10)
            # Initialize a list to store the residual's stddev depending on the minimum standard deviation of the ray
            stddevs = []
            # By varying increasingly the standard deviation minimum of a single gaussian function, the residual's stddev
            # sometimes has a peak before diminishing and then climbing again. The function detects when there has been
            # two "bumps", meaning when the stddev of the residual has risen from its previous value twice, indicating a
            # true minimum.
            stddev_bump_count = 0
            while stddev_bump_count < 2 and len(stddevs) < 40:
                # Set the self.fitted_gaussian variable to allow the calculation of the subtracted fit
                new_gaussian = self.fit(params=initial_guesses, stddev_mins={ray: min_guess})
                stddevs.append(float(self.get_residue_stddev()))
                min_guess += stddev_increments
                try:
                    if stddevs[-1] > stddevs[-2]:
                        stddev_bump_count += 1
                except:
                    continue
            # Store the best standard deviation value found for the corresponding ray
            stddev_mins[ray] = (stddevs.index(min(stddevs)))*stddev_increments + 10 ** (-10)
        # Fit the data with the standard deviation minimums found
        return self.fit(initial_guesses, stddev_mins)
    
    def get_initial_guesses(self, number_of_components: int=6) -> dict:
        """
        Find the most plausible initial guess for the amplitude and mean value of every gaussian function with the NII data cube.

        Arguments
        ---------
        number_of_components: int, default=6. Number of initial guesses that need to be returned. This integer may be 6 or 7
        depending on if a double NII peak is visible.

        Returns
        -------
        dict: to every ray (key) is associated another dict in which the keys are the amplitude "a" and the mean value "x0".
        """
        params = {}
        # Trial and error determined value that allows the best detection of a peak by measuring the difference between
        # consecutive derivatives
        diff_threshold = -0.45
        # Trial and error determined value that acts specifically on the determination of the OH3 peak by looking at the
        # difference between consecutive derivatives in the case that no peak is present
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
        # Note that the first element of the list is the derivative difference between channels 2 and 3 and channels 1 and 2
        
        x_peaks = {}
        for ray, bounds in [("OH1", (1,5)), ("OH2", (19,21)), ("OH3", (36,39)), ("OH4", (47,48)), ("NII", (13,17)), ("Ha", (42,45))]:
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
            # For the OH1 and OH4 rays, the maximum intensity is used as the initial guess
            if ray != "OH1" and ray != "OH4":
                for x in range(bounds[0], bounds[1]):
                    # Variables used to ease the use of lists
                    x_list_deriv = x - 2
                    x_list = x - 1
                    if ray == "OH3":
                        # First condition checks if a significant change in derivative is noticed which could indicate a peak
                        # Also makes sure that the peak is higher than any peak that might have been found previously
                        if derivatives_diff[x_list_deriv] < diff_threshold and (
                            self.y_values[x_list] > self.y_values[x_peak_OH3-1] or x_peak_OH3 == 0):
                            x_peak_OH3 = x

                        # In the case that no peak is noticed, the second condition checks when the derivative suddenly rises
                        # This can indicate a "bump" in the emission ray's shape, betraying the presence of another component
                        # This condition is only True once as the derivatives keep rising after the "bump"
                        if derivatives_diff[x_list_deriv] > diff_threshold_OH3 and not stop_OH3:
                            x_peak = x
                            stop_OH3 = True

                    else:
                        # For other rays, only a significant change in derivative is checked while making sure it is the max value
                        if derivatives_diff[x_list_deriv] < diff_threshold and (
                            self.y_values[x_list] > self.y_values[x_peak-1] or x_peak == 0):
                            x_peak = x

            # If no peak is found, the peak is chosen to be the maximum value within the bounds
            if x_peak == 0:
                x_peak = list(self.y_values[bounds[0]-1:bounds[1]-1]).index(max(self.y_values[bounds[0]-1:bounds[1]-1])) + bounds[0]
            
            # If a real peak has been found for the OH3 ray, it predominates over a big rise in derivative
            if x_peak_OH3 != 0:
                x_peak = x_peak_OH3

            x_peaks[ray] = x_peak
        
        for ray in ["OH1", "OH2", "OH3", "OH4", "NII", "Ha"]:
            params[ray] = {"x0": x_peaks[ray], "a": self.y_values[x_peaks[ray]-1]}
        if number_of_components == 7:
            params["NII_2"] = {"x0": x_peaks["NII_2"], "a": self.y_values[x_peaks["NII_2"]-1]}
        return params
        
    def get_fit_parameters(self, peak_name: str=None) -> models:
        """
        Get the parameters of every gaussian component of the complete fit. Each component is an index of the returned object.

        Arguments
        ---------
        peak_name: str, default=None. Specifies from which peak the function needs to be extracted. The supported peaks are:
        "OH1", "OH2", "OH3", "OH4", "NII", "Ha" and "NII_2". No peak_name needs to be provided in the case of the calibration cube.

        Returns
        -------
        astropy.modeling.core.CompoundModel: function representing the specified peak
        """
        peak_numbers = {"OH1": 0, "OH2": 1, "OH3": 2, "OH4": 3, "NII": 4, "Ha": 5, "NII_2": 6}
        if peak_name is None:
            return self.fitted_gaussian
        return self.fitted_gaussian[peak_numbers[peak_name]]
    
    def get_uncertainties(self) -> dict:
        """
        Get the uncertainty on every parameter of every gaussian component.

        Returns
        -------
        dict: every key is a function ("OH1", "OH2", ...) and the value is another dict with the uncertainty values linked
        to the keys "amplitude", "mean" and "stddev". If the calibration cube was fitted, then the uncertainty is associated
        to the key "calibration".
        """
        cov_matrix = self.fitted_gaussian.meta["fit_info"]["param_cov"]
        uncertainty_matrix = np.sqrt(np.diag(cov_matrix))
        # The uncertainty matrix is stored as a_0, x0_0, sigma_0, a_1, x0_1, sigma_1, ...
        if self.calibration: 
            return {"calibration": {"amplitude": uncertainty_matrix[0], "mean": uncertainty_matrix[1],
                                    "stddev": uncertainty_matrix[2]}}
        ordered_uncertainties = {}
        for i, peak_name in zip(range(int(len(uncertainty_matrix)/3)), (["OH1", "OH2", "OH3", "OH4", "NII", "Ha"])):
            ordered_uncertainties[peak_name] = {
                "amplitude": uncertainty_matrix[3*i], "mean": uncertainty_matrix[3*i+1], "stddev": uncertainty_matrix[3*i+2]
            }
        # Check if the fit was done with seven components
        if len(uncertainty_matrix)/3 == 7:
            ordered_uncertainties["NII_2"] = {
                "amplitude": uncertainty_matrix[3*6], "mean": uncertainty_matrix[3*6+1], "stddev": uncertainty_matrix[3*6+2]
            }
        return ordered_uncertainties
    
    def get_residue_stddev(self) -> float:
        """
        Get the standard deviation of the fit's residue.
        
        Returns
        -------
        float: value of the residue's standard deviation.
        """
        return np.std(self.get_subtracted_fit()/u.Jy)
    
    def get_subtracted_fit(self) -> np.ndarray:
        """
        Get the values of the subtracted_fit.

        Returns
        -------
        numpy array: gaussian fit subtracted to the y values.
        """
        subtracted_y = self.y_values*u.Jy - self.fitted_gaussian(self.x_values*u.um)
        return subtracted_y
    
    def get_FWHM_channels(self, function, stddev_uncertainty: float) -> np.ndarray:
        """
        Get the full width at half max of a function along with its uncertainty in channels.

        Arguments
        ---------
        function: astropy.modeling.core.CompoundModel. Specifies the gaussian function whose FWHM must be computed.
        stddev_uncertainty: float. Corresponds to the function's standard deviation uncertainty.

        Returns
        -------
        numpy array: array of the FWHM and its uncertainty measured in channels.
        """
        fwhm = 2*np.sqrt(2*np.log(2))*function.stddev.value 
        fwhm_uncertainty = 2*np.sqrt(2*np.log(2))*stddev_uncertainty
        return np.array((fwhm, fwhm_uncertainty))

    def get_FWHM_speed(self, peak_name: str=None) -> np.ndarray:
        """
        Get the full width at half max of a function along with its uncertainty in km/s.

        Arguments
        ---------
        peak_name: str default=None. Name of the peak whose FWHM in km/s is desired. The supported peaks are:
        "OH1", "OH2", "OH3", "OH4", "NII" and "Ha". In the case of the calibration cube, no peak needs to be provided.

        Returns
        -------
        numpy array: array of the FWHM and its uncertainty measured in km/s.
        """
        # The two following values are provided in the cube's header
        spectral_length = self.header["FP_I_A"]
        wavelength_channel_1 = self.header["FP_B_L"]
        number_of_channels = self.header["NAXIS3"]
        if peak_name is not None:
            # A multi-gaussian fit was done
            params = self.get_fit_parameters(peak_name)
            uncertainties = self.get_uncertainties()[peak_name]
            channels_FWHM = self.get_FWHM_channels(params, uncertainties["stddev"])
            angstroms_center = (np.array((params.mean.value, uncertainties["mean"])) * spectral_length / number_of_channels)
        else:
            # A single gaussian fit was done
            params = self.get_fit_parameters()
            uncertainties = self.get_uncertainties()["calibration"]
            channels_FWHM = self.get_FWHM_channels(params, uncertainties["stddev"])
            angstroms_center = (np.array((params.mean.value, uncertainties["mean"])) * spectral_length / number_of_channels)
        angstroms_center[0] +=  wavelength_channel_1
        angstroms_FWHM = channels_FWHM * spectral_length / number_of_channels
        speed_FWHM = scipy.constants.c * angstroms_FWHM[0] / angstroms_center[0] / 1000
        speed_FWHM_uncertainty = (scipy.constants.c / 1000 * 
            (angstroms_FWHM[1]/angstroms_FWHM[0] + angstroms_center[1]/angstroms_center[0]) * angstroms_FWHM[0] / angstroms_center[0])
        speed_array = np.array((speed_FWHM, speed_FWHM_uncertainty))
        # Check if the NII peak is used and if a double fit was done
        if peak_name == "NII":
            try:
                return np.mean((speed_array, self.get_FWHM_speed("NII_2")), axis=0)
            except:
                pass
        return speed_array
    
    def get_snr(self, peak_name: str) -> float:
        """
        Get the signal to noise ratio of a peak. This is calculated as the amplitude of the peak divided by the residue's standard
        deviation.

        Arguments
        ---------
        peak_name: str. Name of the peak whose amplitude will be used to calculate the snr.

        Returns
        -------
        float: value of the signal to noise ratio.
        """
        return (self.get_fit_parameters(peak_name).amplitude/u.Jy)/self.get_residue_stddev()
    
    def get_FWHM_snr_7_components_array(self):
        """
        Get the 7x3 dimensional array representing the FWHM, snr and 7 components fit. This method is used in the worker_fit()
        function which creates heavy arrays.
        
        Returns
        -------
        Numpy array: all values are specific to a certain pixel that was fitted. For the first six rows, the first element is
        the FWHM value in km/s, the second element is the uncertainty in km/s and the third element is the snr of the peak. The
        peaks are in the following order: OH1, OH2, OH3, OH4, NII and Ha. The last row has only a relevant element in the first
        column: it takes the value 1 if a double NII peak was considered and 0 otherwise. The two other rows are filled with 
        False only to make the array have the same length in every dimension.
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
    
    def get_amplitude_7_components_array(self):
        """
        Get the 7x3 dimensional array representing the amplitude and 7 components fit. This method is used in the worker_fit()
        function which creates heavy arrays.
        
        Returns
        -------
        Numpy array: all values are specific to a certain pixel that was fitted. For the first six rows, the first element is
        the amplitude value, the second element is the uncertainty and a False, present to make the array have the same shape
        then the array given by the get_FWHM_snr_7_components_array() method. The peaks are in the following order: OH1, OH2,
        OH3, OH4, NII and Ha. The last row has only a relevant element in the first column: it takes the value 1 if a double
        NII peak was considered and 0 otherwise. The two other rows are filled with False only to make the array have the same
        length in every dimension.
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
                [np.nanmean((self.get_fit_parameters("NII").amplitude.value, self.get_fit_parameters("NII_2").amplitude.value)),
                 np.nanmean((self.get_uncertainties()["NII"]["amplitude"], self.get_uncertainties()["NII_2"]["amplitude"])), 
                 False],
                [self.get_fit_parameters("Ha").amplitude.value, self.get_uncertainties()["Ha"]["amplitude"], False],
                [self.seven_components_fit, False, False]
            ))
        
    def get_mean_7_components_array(self):
        """
        Get the 7x3 dimensional array representing the mean and 7 components fit. This method is used in the worker_fit()
        function which creates heavy arrays.
        
        Returns
        -------
        Numpy array: all values are specific to a certain pixel that was fitted. For the first six rows, the first element is
        the mean value, the second element is the uncertainty and a False, present to make the array have the same shape
        then the array given by the get_FWHM_snr_7_components_array() method. The peaks are in the following order: OH1, OH2,
        OH3, OH4, NII and Ha. The last row has only a relevant element in the first column: it takes the value 1 if a double
        NII peak was considered and 0 otherwise. The two other rows are filled with False only to make the array have the same
        length in every dimension.
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
                [np.nanmean((self.get_fit_parameters("NII").mean.value, self.get_fit_parameters("NII_2").mean.value)), 
                 np.nanmean((self.get_uncertainties()["NII"]["mean"], self.get_uncertainties()["NII_2"]["mean"])), 
                 False],
                [self.get_fit_parameters("Ha").mean.value, self.get_uncertainties()["Ha"]["mean"], False],
                [self.seven_components_fit, False, False]
            ))

""" 
def loop_di_loop(filename, calib=False):
    x = 300
    iter = open("gaussian_fitting/other/iter_number.txt", "r").read()
    for y in range(int(iter), 1013):
        print(f"\n----------------\ncoords: {x,y}")
        data = fits.open(filename)[0].data
        header = fits.open(filename)[0].header
        spectrum = Spectrum(data[:,y-1,x-1], header, calibration=calib)
        spectrum.fit_NII_cube()
        # spectrum.fit_iteratively()
        # print("FWHM NII:", spectrum.get_FWHM_speed(spectrum.get_fit_parameters()[4], spectrum.get_uncertainties()["g4"]["stddev"]))
        # print("FWHM Ha:", spectrum.get_FWHM_speed(spectrum.get_fit_parameters()[5], spectrum.get_uncertainties()["g5"]["stddev"]))
        # print("standard deviation:", spectrum.get_residue_stddev())
        print(spectrum.fitted_gaussian)
        try:
            print("mean FWHM:", spectrum.get_FWHM_speed("NII"))
        except:
            try:
                print("mean FWHM:", (spectrum.get_FWHM_speed()))
            except:
                pass
        # spectrum.plot(coords=(x,y))
        # M0 = np.sum(spectrum.get_fit_parameters('NII')(np.linspace(1,48,48)*u.um)*header['CDELT3']*u.um)
        # print(f"M0: {M0}")
        # M1 = np.sum(spectrum.get_fit_parameters('NII')(np.linspace(1,48,48)*u.um)*header['CDELT3']*u.um*np.linspace(header["CRVAL3"], header["CRVAL3"]+header["CDELT3"]*47, 48)) / M0
        # print(f"M1: {M1}")
        spectrum.plot_fit(fullscreen=False, coords=(x,y), plot_all=True)
        file = open("gaussian_fitting/other/iter_number.txt", "w")
        file.write(str(y+1))
        file.close()
loop_di_loop("gaussian_fitting/data_cubes/night_34_binned.fits")
# loop_di_loop("gaussian_fitting/leo/OIII/reference_cube_with_header.fits", calib=True)
 """