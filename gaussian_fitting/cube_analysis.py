import matplotlib.pyplot as plt
import numpy as np

from astropy.modeling import models, fitting
from astropy.io import fits
from astropy import units as u

from scipy.optimize import fsolve

from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines

class Spectrum:

    def __init__(self, data=np.ndarray, calibration=False, desired_peak_position=35):
        """
        Calibration boolean must be set to True to force the analysis of a single peak.
        """
        self.data = data
        self.x_values, self.y_values = np.arange(48) + 1, data
        self.calibration = calibration
        
        if calibration:
            desired_peak_position = 35
            temporary_max_x = list(self.y_values).index(max(self.y_values))
            peak_position_translation = desired_peak_position - temporary_max_x
            new_y_values = np.zeros(shape=48)

            for i, value in enumerate(self.y_values):
                if (peak_position_translation + i) >= 48:
                    new_y_values[i+peak_position_translation-48] = value
                else:
                    new_y_values[i+peak_position_translation] = value

            self.old_y_values = self.y_values
            self.y_values = new_y_values
            max_intensity_x = list(self.y_values).index(max(self.y_values))
            mean = np.sum(self.y_values[0:25]) / 25
            self.y_values -= mean
            self.max_tuple = (int(self.x_values[max_intensity_x]), float(self.y_values[max_intensity_x]))

        else:
            mean = np.sum(self.y_values[24:34]) / 10
            self.y_values -= mean
        
    def plot(self, coords, fullscreen=False, **other_values):
        fig, axs = plt.subplots(2)
        for name, value in other_values.items():
            # For neat gaussian functions
            x_plot_gaussian = np.arange(1,48.05,0.05)
            if name == "fit":
                # Fitted entire function
                axs[0].plot(x_plot_gaussian*u.Jy, value(x_plot_gaussian*u.um), "r-", label=name)
            elif name == "subtracted_fit":
                # Residual distribution
                axs[1].plot(self.x_values, value, label=name)
            elif name == "NII":
                axs[0].plot(x_plot_gaussian, value(x_plot_gaussian), "m-", label=name, linewidth="1")
            else:
                # Fitted individual gaussians
                axs[0].plot(x_plot_gaussian, value(x_plot_gaussian), "y-", label=name, linewidth="1")
        
        axs[0].plot(self.x_values, self.y_values, "g-", label="ds9 spectrum", linewidth=3, alpha=0.6)
        axs[0].legend(loc="upper left", fontsize="7")
        axs[1].legend(loc="upper left", fontsize="7")
        plt.xlabel("channels")
        axs[0].set_ylabel("intensity")
        axs[1].set_ylabel("intensity")
        """-----------------------------------------"""
        fig.text(0.4, 0.89, f"coords: {coords}, stddev: {self.get_stddev(self.get_subtracted_fit())}")
        fig.text(0.02, 0.96, self.peaks, fontsize=9.8)
        """-----------------------------------------"""
        if fullscreen:    
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        plt.show()

    def plot_fit(self, coord, fullscreen=False, plot_all=False):
        if plot_all and not self.calibration:
            g = self.fitted_gaussian
            oh1 = models.Gaussian1D(amplitude=g.amplitude_0.value, mean=g.mean_0.value, stddev=g.stddev_0.value)
            oh2 = models.Gaussian1D(amplitude=g.amplitude_1.value, mean=g.mean_1.value, stddev=g.stddev_1.value)
            oh3 = models.Gaussian1D(amplitude=g.amplitude_2.value, mean=g.mean_2.value, stddev=g.stddev_2.value)
            oh4 = models.Gaussian1D(amplitude=g.amplitude_3.value, mean=g.mean_3.value, stddev=g.stddev_3.value)
            nii = models.Gaussian1D(amplitude=g.amplitude_4.value, mean=g.mean_4.value, stddev=g.stddev_4.value)
            ha  = models.Gaussian1D(amplitude=g.amplitude_5.value, mean=g.mean_5.value, stddev=g.stddev_5.value)
            self.plot(coord, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit(),
                      OH1=oh1, OH2=oh2, OH3=oh3, OH4=oh4, NII=nii, Ha=ha)
        
        else:
            self.plot(coord, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit())

    def fit(self):
        spectrum = Spectrum1D(flux=self.y_values*u.Jy, spectral_axis=self.x_values*u.um)
        if self.calibration:
            g_init = models.Gaussian1D(amplitude=self.max_tuple[1]*u.Jy, mean=self.max_tuple[0]*u.um)

            self.fitted_gaussian = fit_lines(spectrum, g_init,
                                                fitter=fitting.LMLSQFitter(calc_uncertainties=True), get_fit_info=True)

        else:
            params = self.get_initial_guesses()
            # Initialize the Gaussians
            g_init_OH1 = models.Gaussian1D(amplitude=params["OH1"]["a"]*u.Jy, mean=params["OH1"]["x0"]*u.um, 
                                           bounds={"amplitude": (0,100)*u.Jy})
            g_init_OH2 = models.Gaussian1D(amplitude=params["OH2"]["a"]*u.Jy, mean=params["OH2"]["x0"]*u.um, 
                                           bounds={"amplitude": (0,100)*u.Jy, "mean": (17,21)})
            g_init_OH3 = models.Gaussian1D(amplitude=params["OH3"]["a"]*u.Jy, mean=params["OH3"]["x0"]*u.um, 
                                           bounds={"amplitude": (0,100)*u.Jy, "mean": (36,42)})
            g_init_OH4 = models.Gaussian1D(amplitude=params["OH4"]["a"]*u.Jy, mean=params["OH4"]["x0"]*u.um, 
                                           bounds={"amplitude": (0,100)*u.Jy})
            g_init_NII = models.Gaussian1D(amplitude=params["NII"]["a"]*u.Jy, mean=params["NII"]["x0"]*u.um,
                                           bounds={"amplitude": (0,100)*u.Jy, "mean": (12,16)})
            g_init_Ha  = models.Gaussian1D(amplitude=params["Ha"]["a"]*u.Jy,  mean=params["Ha"]["x0"]*u.um,
                                           bounds={"amplitude": (0,100)*u.Jy, "mean": (41,45)})
            g_init_OH1.mean.max = 4
            g_init_OH4.mean.min = 47

            self.fitted_gaussian = fit_lines(spectrum, g_init_OH1 + g_init_OH2 + g_init_OH3 + g_init_OH4 + g_init_NII + g_init_Ha,
                                                fitter=fitting.LMLSQFitter(calc_uncertainties=True), get_fit_info=True)

    def get_initial_guesses(self):
        # Outputs a dict of every peak and the a and x0 initial guesses
        params = {}
        diff_threshold = -0.45
        diff_threshold_OH3 = 1.8

        derivatives = np.zeros(shape=(47,2))
        for i in range(0, len(self.x_values)-1):
            derivatives[i,0] = i + 1
            derivatives[i,1] = self.y_values[i+1] - self.y_values[i]

        # The first element is the derivative difference at point x = 2.
        derivatives_diff = []
        for x in range(2,48):
            x_list = x - 1
            derivatives_diff.append(derivatives[x_list,1] - derivatives[x_list-1,1])
        
        x_peaks = {"OH1": list(self.y_values[0:4]).index(max(self.y_values[0:4])) + 1}
        for ray, bounds in [("OH2", (18,21)), ("OH3", (36,40)), ("OH4", (47,48)), ("NII", (13,16)), ("Ha", (42,45))]:
            x_peak = 0
            x_peak_OH3 = 0
            stop_OH3 = False
            
            if ray != "OH4":
                for x in range(bounds[0], bounds[1]):
                    x_list_deriv = x - 2
                    x_list = x - 1
                    if ray == "OH3":
                        if derivatives_diff[x_list_deriv] < diff_threshold and (
                            self.y_values[x_list] > self.y_values[x_peak_OH3-1] or x_peak_OH3 == 0):
                            x_peak_OH3 = x

                        if derivatives_diff[x_list_deriv] > diff_threshold_OH3 and not stop_OH3:
                            x_peak = x
                            stop_OH3 = True

                    else:
                        if derivatives_diff[x_list_deriv] < diff_threshold and (
                            self.y_values[x_list] > self.y_values[x_peak-1] or x_peak == 0):
                            x_peak = x

            if x_peak == 0:
                x_peak = list(self.y_values[bounds[0]-1:bounds[1]-1]).index(max(self.y_values[bounds[0]-1:bounds[1]-1])) + bounds[0]
            
            if x_peak_OH3 != 0:
                x_peak = x_peak_OH3

            x_peaks[ray] = x_peak
        
        for ray in ["OH1", "OH2", "OH3", "OH4", "NII", "Ha"]:
            params[ray] = {"x0": x_peaks[ray], "a": self.y_values[x_peaks[ray]-1]}
        
        self.peaks = params
        return params
    
    def get_fitted_gaussian_parameters(self):
        return self.fitted_gaussian
    
    def get_uncertainties(self):
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
        return np.std(array)
        
    def get_subtracted_fit(self):
        subtracted_y = self.y_values*u.Jy - self.fitted_gaussian(self.x_values*u.um)
        return subtracted_y
    
    def get_FWHM(self, function, stddev_uncertainty):
        fwhm = 2*np.sqrt(2*np.log(2))*function.stddev.value 
        fwhm_uncertainty = 2*np.sqrt(2*np.log(2))*stddev_uncertainty
        return [fwhm, fwhm_uncertainty]


def loop_di_loop(filename):
    calib = False
    if filename == "calibration.fits":
        calib = True
    x = 140
    for y in range(150, 300):
        print(f"\n----------------\ncoords: {x,y}")
        data = fits.open(filename)[0].data
        spectrum = Spectrum(data[:,y-1,x-1], calibration=calib)
        spectrum.fit()
        print(spectrum.get_FWHM(spectrum.fitted_gaussian[4], spectrum.get_uncertainties()["g4"]["stddev"]))
        spectrum.plot_fit(fullscreen=True, coord=(x,y), plot_all=True)

loop_di_loop("cube_NII_Sh158_with_header.fits")
# loop_di_loop("calibration.fits")
