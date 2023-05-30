
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

from astropy.modeling import models, fitting
from astropy.io import fits
from astropy import units as u

from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines


class Spectrum:
    """
    Encapsulate all the data and methods of a cube's spectrum.
    """

    def __init__(self, data=np.ndarray, calibration=bool, desired_peak_position=35):
        """
        Initialize a Spectrum object. Calibration boolean must be set to True to force the analysis of a single peak.
        
        Arguments
        ---------
        data: numpy array of the flux at different channels.
        calibration: bool specifying if the fit is for the calibration cube i.e. to fit a single peak. If False, the fitter will
        attempt a 6 components fit.
        desired_peak_position: int that specifies the location of the single peak for the calibration cube. All values will be
        shifted accordingly.
        """
        self.x_values, self.y_values = np.arange(48) + 1, data
        self.calibration = calibration
        self.data = data

        if calibration:
            # Application of a translation in the case of the calibration cube
            # The distance between the desired peak and the current peak is calculated
            peak_position_translation = desired_peak_position - (list(self.y_values).index(max(self.y_values)) + 1)
            self.y_values = np.roll(self.y_values, peak_position_translation)
            # All y values are shifted downwards by the mean calculated in the 25 first channels
            mean = np.sum(self.y_values[0:25]) / 25
            self.y_values -= mean
            # A tuple containing the peak's x and y is stored
            self.max_tuple = (int(self.x_values[desired_peak_position - 1]), float(self.y_values[desired_peak_position - 1]))

        else:
            # All y values are shifted downwards by the mean calculated in the channels 25 to 35
            mean = np.sum(self.y_values[24:34]) / 10
            self.y_values -= mean
        
    def plot(self, coords=None, fullscreen=False, **other_values):
        """
        Plot the data and the fits.
        
        Arguments
        ---------
        coords: optional tuple of the x and y coordinates of the evaluated point. Serves as a landmark in the cube and will
        appear on screen.
        fullscreen: boolean that specifies if the graph must be opened in fullscreen.
        other_values: optional argument that may take any distribution to be plotted. This argument is used to plot all
        the gaussian fits.
        """
        fig, axs = plt.subplots(2)
        # Plot of the data
        axs[0].plot(self.x_values, self.y_values, "g-", label="ds9 spectrum", linewidth=3, alpha=0.6)
        for name, value in other_values.items():
            x_plot_gaussian = np.arange(1,48.05,0.05)
            if name == "fit":
                # Fitted entire function
                axs[0].plot(x_plot_gaussian*u.Jy, value(x_plot_gaussian*u.um), "r-", label=name)
            elif name == "subtracted_fit":
                # Residual distribution
                axs[1].plot(self.x_values, value, label=name)
            elif name == "NII":
                # NII gaussian
                axs[0].plot(x_plot_gaussian, value(x_plot_gaussian), "m-", label=name, linewidth="1")
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

    def plot_fit(self, coords=None, fullscreen=False, plot_all=False):
        """
        Send all the functions to be plotted to the plot method depending on the data cube used.

        Arguments
        ---------
        coord: tuple of the x and y coordinates of the evaluated point. Serves as a landmark in the cube and will
        appear on screen.
        fullscreen: boolean that specifies if the graph must be opened in fullscreen.
        plot_all: boolean that specifies if all gaussian functions contributing to the main fit must be plotted.
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
            self.plot(coords, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit(),
                      OH1=oh1, OH2=oh2, OH3=oh3, OH4=oh4, NII=nii, Ha=ha)
        
        else:
            self.plot(coords, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit())

    def fit(self, params=None, stddev_mins=None):
        """
        Fit the data using specutils and initial guesses.

        Arguments
        ---------
        params: optional dict containing the initial guesses for the amplitude and mean of each gaussian component. In the
        case of the calibration cube, the initial guesses are defined within the function and no dict is needed.
        stddev_mins: optional dict that specifies the standard deviation's minimum value of every gaussian component.
        This is used in the fit_iteratively method to increase the fit's accuracy.

        Returns
        -------
        astropy.modeling.core.CompoundModel: model of the fitted distribution using 6 gauss functions (only one for
        the calibration cube). Also sets the astropy model to the variable self.fitted_gaussian.
        """
        spectrum = Spectrum1D(flux=self.y_values*u.Jy, spectral_axis=self.x_values*u.um)
        if self.calibration:
            # Initialize the single gaussian using the max peak's position
            g_init = models.Gaussian1D(amplitude=self.max_tuple[1]*u.Jy, mean=self.max_tuple[0]*u.um)

            self.fitted_gaussian = fit_lines(spectrum, g_init,
                                                fitter=fitting.LMLSQFitter(calc_uncertainties=True), get_fit_info=True)

        else:
            # Initialize the six gaussians using the params dict
            g_init_OH1 = models.Gaussian1D(amplitude=params["OH1"]["a"]*u.Jy, mean=params["OH1"]["x0"]*u.um, 
                                           bounds={"amplitude": (0,100)*u.Jy})
            g_init_OH2 = models.Gaussian1D(amplitude=params["OH2"]["a"]*u.Jy, mean=params["OH2"]["x0"]*u.um,
                                           bounds={"amplitude": (0,100)*u.Jy, "mean": (17,21)*u.um})
            g_init_OH3 = models.Gaussian1D(amplitude=params["OH3"]["a"]*u.Jy, mean=params["OH3"]["x0"]*u.um, 
                                           bounds={"amplitude": (0,100)*u.Jy, "mean": (36,40)*u.um})
            g_init_OH4 = models.Gaussian1D(amplitude=params["OH4"]["a"]*u.Jy, mean=params["OH4"]["x0"]*u.um, 
                                           bounds={"amplitude": (0,100)*u.Jy})
            g_init_NII = models.Gaussian1D(amplitude=params["NII"]["a"]*u.Jy, mean=params["NII"]["x0"]*u.um,
                                           bounds={"amplitude": (0,100)*u.Jy, "mean": (12,16)*u.um})
            g_init_Ha  = models.Gaussian1D(amplitude=params["Ha"]["a"] *u.Jy, mean=params["Ha"]["x0"] *u.um,
                                           bounds={"amplitude": (0,100)*u.Jy, "mean": (41,45)*u.um})
            g_init_OH1.mean.max = 3*u.um
            g_init_OH4.mean.min = 47*u.um
            
            # Set the standard deviation's minimum of the gaussians of the corresponding rays if the dict is present
            if stddev_mins:
                for ray, min_guess in stddev_mins.items():
                    exec(f"g_init_{ray}.stddev.min = {min_guess}*u.um")

            self.fitted_gaussian = fit_lines(spectrum, g_init_OH1 + g_init_OH2 + g_init_OH3 + g_init_OH4 + g_init_NII + g_init_Ha,
                                                fitter=fitting.LMLSQFitter(calc_uncertainties=True), get_fit_info=True)
            return self.fitted_gaussian

    def fit_iteratively(self, stddev_increments=0.2):
        """
        Use the fit method iteratively to find the best possible standard deviation values for the gauss functions representing
        the OH emission rays by minimizing the residual's standard deviation. After finding every best minimum standard
        deviation value, a fit is made with those values.

        Arguments
        ---------
        stddev_increments: float that indicates the increments that will be used to test every standard deviation value for
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
            while stddev_bump_count < 2:
                # Set the self.fitted_gaussian variable to allow the calculation of the subtracted fit
                new_gaussian = self.fit(params=initial_guesses, stddev_mins={ray: min_guess})
                stddevs.append(float(self.get_stddev(self.get_subtracted_fit()/u.Jy)))
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

    def get_initial_guesses(self):
        """
        Find the most plausible initial guess for the amplitude and mean value of every gaussian function with the NII data cube.

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
        for ray, bounds in [("OH1", (1,5)), ("OH2", (18,21)), ("OH3", (36,40)), ("OH4", (47,48)), ("NII", (13,16)), ("Ha", (42,45))]:
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
                        # First condition checks if a significant change in dervative is noticed which could indicate a peak
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
        return params
    
    def get_fitted_gaussian_parameters(self):
        """
        Get the parameters of every gaussian component of the complete fit.

        Returns
        -------
        astropy.modeling.core.CompoundModel: may be printed to see the value of every parameter.
        """
        return self.fitted_gaussian
    
    def get_uncertainties(self):
        """
        Get the uncertainty on every parameter of every gaussian component.

        Returns
        -------
        dict: every key is a function ("g0", "g1", ...) and the value is another dict with the uncertainty values linked to the
        keys "amplitude", "mean" and "stddev". Note that the gaussian representing the NII peak is labeled 'g4'.
        """
        cov_matrix = self.fitted_gaussian.meta["fit_info"]["param_cov"]
        uncertainty_matrix = np.sqrt(np.diag(cov_matrix))
        # The uncertainty matrix is stored as a_0, x0_0, sigma_0, a_1, x0_1, sigma_1, ...
        ordered_uncertainties = {}
        for i in range(int(len(uncertainty_matrix)/3)):
            ordered_uncertainties[f"g{i}"] = {
                "amplitude": uncertainty_matrix[3*i], "mean": uncertainty_matrix[3*i+1], "stddev": uncertainty_matrix[3*i+2]
            }
        return ordered_uncertainties 
    
    def get_stddev(self, array):
        """
        Get the standard deviation of any array. It is mainly used to find the subtracted fit's standard deviation.

        Returns
        -------
        float: value of the array's standard deviation.
        """
        return np.std(array)
        
    def get_subtracted_fit(self):
        """
        Get the values of the subtracted_fit.

        Returns
        -------
        numpy array: gaussian fit subtracted to the y values.
        """
        subtracted_y = self.y_values*u.Jy - self.fitted_gaussian(self.x_values*u.um)
        return subtracted_y
    
    def get_FWHM_channels(self, function, stddev_uncertainty):
        """
        Get the full width at half max of a function along with its uncertainty in channels.

        Arguments
        ---------
        function: astropy.modeling.core.CompoundModel that specifies the gaussian function whose FWHM must be computed.
        stddev_uncertainty: float corresponding to the function's standard deviation uncertainty.

        Returns
        -------
        numpy array: array of the FWHM and its uncertainty measured in channels.
        """
        fwhm = 2*np.sqrt(2*np.log(2))*function.stddev.value 
        fwhm_uncertainty = 2*np.sqrt(2*np.log(2))*stddev_uncertainty
        return np.array((fwhm, fwhm_uncertainty))

    def get_FWHM_speed(self, function, stddev_uncertainty):
        """
        Get the full width at half max of a function along with its uncertainty in km/s.

        Arguments
        ---------
        function: astropy.modeling.core.CompoundModel that specifies the gaussian function whose FWHM must be computed.
        stddev_uncertainty: float corresponding to the uncertainty of the function's standard deviation.

        Returns
        -------
        numpy array: array of the FWHM and its uncertainty measured in km/s.
        """
        spectral_length = 8.60626405229
        wavelength_channel_1 = 6579.48886797
        channels_FWHM = self.get_FWHM_channels(function, stddev_uncertainty)
        angstroms_FWHM = channels_FWHM * spectral_length / 48
        angstroms_center = function.mean.value * spectral_length / 48 + wavelength_channel_1
        speed_FWHM = scipy.constants.c * angstroms_FWHM / angstroms_center / 1000
        return speed_FWHM


"""
def loop_di_loop(filename):
    calib = False
    if filename == "calibration.fits":
        calib = True
    x = 527
    for y in range(784, 1000):
        print(f"\n----------------\ncoords: {x,y}")
        data = fits.open(filename)[0].data
        spectrum = Spectrum(data[:,y-1,x-1], calibration=calib)
        spectrum.fit(spectrum.get_initial_guesses())
        # start = time.time()
        # spectrum.fit_iteratively(0.2)
        # stop = time.time()
        # print("time:", stop-start)
        print("FWHM:", spectrum.get_FWHM_speed(spectrum.fitted_gaussian, spectrum.get_uncertainties()["g0"]["stddev"]))
        print("stddev:", spectrum.get_stddev(spectrum.get_subtracted_fit()))
        spectrum.plot_fit(fullscreen=False, coords=(x,y), plot_all=True)

# loop_di_loop("cube_NII_Sh158_with_header.fits")
loop_di_loop("calibration.fits")
"""