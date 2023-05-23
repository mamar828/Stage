import os

import matplotlib.pyplot as plt
import numpy as np

from astropy.modeling import models, fitting
from astropy.io import fits
from scipy.optimize import fsolve

from astropy import units as u

from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines

class Spectrum:

    def __init__(self, data=np.ndarray, desired_peak_position=35):
        self.data = data

        try:
            self.x_values, self.y_values = np.split(data, 2, axis=1)
        except Exception:
            self.x_values, self.y_values = np.arange(48) + 1, data
        
        mean = np.sum(self.y_values[24:34]) / 10
        self.y_values -= mean
        
        
    def plot(self, coords, fullscreen=False, **other_values):
        fig, axs = plt.subplots(2)
        for name, value in other_values.items():
            try:
                # For neat gaussian function
                x_plot = np.arange(1,49,0.05)
                if name == "fit":
                    axs[0].plot(x_plot*u.Jy, value(x_plot*u.um), "r-", label=name)
                else:
                    axs[0].plot(x_plot, value(x_plot), "y-", label=name, linewidth="1")
            except Exception:
                try:
                    # For function evaluated at the same x_values
                    if name == "subtracted_fit":
                        axs[1].plot(self.x_values, value, label=name)
                    else:
                        plt.plot(self.x_values, value, label=name)
                except Exception:
                    # For few points
                    plt.plot(value[:,0], value[:,1], "og", label=name)
            
        axs[0].plot(self.x_values, self.y_values, "g-", label="ds9 spectrum", linewidth=3, alpha=0.6)
        axs[0].legend(loc="upper left", fontsize="8")
        axs[1].legend(loc="upper left", fontsize="8")
        plt.xlabel("channels")
        axs[0].set_ylabel("intensity")
        axs[1].set_ylabel("intensity")
        # print("----------------------- uncertainties -----------------------\n",self.get_uncertainties())
        print(self.get_fitted_gaussian_parameters())
        # print("stddev:", self.get_stddev(self.get_subtracted_fit()))
        # print(self.get_FWHM(self.fitted_gaussian[4]))
        fig.text(0.4, 0.92, f"coords: {coords}, stddev: {self.get_stddev(self.get_subtracted_fit())}")
        fig.text(0.02, 0.96, self.peaks, fontsize=9.8)
        if fullscreen:    
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        plt.show()

    def plot_fit(self, coord, fullscreen=False, plot_all=False):
        if plot_all:
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

    @models.custom_model
    def gauss_function(x, a=1., x0=1., sigma=1., h=0.):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))+h

    def fit_NII(self):
        params = self.get_initial_guesses()
        # Initialize the Gaussians
        g_init_OH1 = models.Gaussian1D(amplitude=params["OH1"]["a"]*u.Jy, mean=params["OH1"]["x0"]*u.um, bounds={"amplitude": (0,100)*u.Jy})
        g_init_OH2 = models.Gaussian1D(amplitude=params["OH2"]["a"]*u.Jy, mean=params["OH2"]["x0"]*u.um, bounds={"amplitude": (0,100)*u.Jy})
        g_init_OH3 = models.Gaussian1D(amplitude=params["OH3"]["a"]*u.Jy, mean=params["OH3"]["x0"]*u.um, bounds={"amplitude": (0,100)*u.Jy})
        g_init_OH4 = models.Gaussian1D(amplitude=params["OH4"]["a"]*u.Jy, mean=params["OH4"]["x0"]*u.um, bounds={"amplitude": (0,100)*u.Jy})
        g_init_NII = models.Gaussian1D(amplitude=params["NII"]["a"]*u.Jy, mean=params["NII"]["x0"]*u.um, bounds={"amplitude": (0,100)*u.Jy})
        g_init_Ha  = models.Gaussian1D(amplitude=params["Ha"]["a"]*u.Jy,  mean=params["Ha"]["x0"]*u.um,  bounds={"amplitude": (0,100)*u.Jy})
                
        g_init_OH1.mean.max = 4
        g_init_OH4.mean.min = 47

        # gaussian_addition_init = g_init_OH1 + g_init_OH2 + g_init_OH3 + g_init_OH4 + g_init_NII + g_init_Ha
        # y_values_fitted = gaussian_addition_init(np.arange(1,49,0.05))

        # gaussian_spectrum = Spectrum1D(flux=y_values_fitted*u.Jy, spectral_axis=np.arange(1,49,0.05)*u.um)
        # fit_g = fit_lines(gaussian_spectrum, gaussian_addition_init)
        # y_fit = fit_g(x*u.um)

        # plt.plot(np.arange(1,49,0.05), self.y_values)
        # plt.plot(np.arange(1,49,0.05), y_fit)
        # plt.title('Double Peak Fit')
        # plt.grid(True)

        
        # self.fit_g = fitting.LMLSQFitter(calc_uncertainties=True)
        # self.fitted_gaussian = self.fit_g(gaussian_addition_init, self.x_values, self.y_values)
        
        
        # Create a simple spectrum with a Gaussian.
        # g1 = models.Gaussian1D(1, 4.6, 0.2)
        # g2 = models.Gaussian1D(2.5, 5.5, 0.1)

        # Create the spectrum to fit
        spectrum = Spectrum1D(flux=self.y_values*u.Jy, spectral_axis=self.x_values*u.um)
        g_123456_init = g_init_OH1+g_init_OH2+g_init_OH3+g_init_OH4+g_init_NII+g_init_Ha
        # print(estimate_line_parameters(spectrum, g_model))
        # Fit the spectrum
        # g1_init = models.Gaussian1D(amplitude=2.3*u.Jy, mean=5.6*u.um, stddev=0.1*u.um)
        # g2_init = models.Gaussian1D(amplitude=1.*u.Jy, mean=4.4*u.um, stddev=0.1*u.um)
        self.fitted_gaussian = fit_lines(spectrum, g_123456_init)
        # y_fit = g12_fit(x*u.um)

        # plt.plot(self.x_values, self.y_values)
        # plt.plot(x, y_fit)
        # plt.title('6 Peak Fit')
        # plt.grid(True)
        # plt.show()

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
        for ray, bounds in [("OH2", (18,22)), ("OH3", (36,40)), ("OH4", (47,48)), ("NII", (13,16)), ("Ha", (42,45))]:
            x_peak = 0
            
            if ray != "OH4":
                for x in range(bounds[0], bounds[1]):
                    x_list_deriv = x - 2
                    x_list = x - 1
                    if ray == "OH3":
                        if derivatives_diff[x_list_deriv] > diff_threshold_OH3:
                            x_peak = x
                            break

                    else:
                        if derivatives_diff[x_list_deriv] < diff_threshold and (
                            self.y_values[x_list] > self.y_values[x_peak-1] or x_peak == 0):
                            x_peak = x

            if x_peak == 0:
                x_peak = list(self.y_values[bounds[0]-1:bounds[1]-1]).index(max(self.y_values[bounds[0]-1:bounds[1]-1])) + bounds[0]
            
            x_peaks[ray] = x_peak
        
        for ray in ["OH1", "OH2", "OH3", "OH4", "NII", "Ha"]:
            params[ray] = {"x0": x_peaks[ray], "a": self.y_values[x_peaks[ray]-1]}
        
        self.peaks = params
        return params
    
    def get_fitted_gaussian_parameters(self):
        return self.fitted_gaussian
    
    def get_uncertainties(self):
        cov_matrix = self.fit_g.fit_info["param_cov"]
        return np.sqrt(np.diag(cov_matrix))
    
    def get_stddev(self, array):
        return np.std(array)
        
    def get_subtracted_fit(self):
        subtracted_y = self.y_values*u.Jy - self.fitted_gaussian(self.x_values*u.um)
        return subtracted_y
    
    def get_FWHM(self, function):
        x = np.arange(1,49,0.01)
        mid_height = max(function(x*u.um))/2

        def gauss_function_intersection(xy):
            x, y = xy
            z = np.array([y - (function.amplitude.value*np.exp(
                -(x*u.um-function.mean.value)**2/(2*function.stddev.value**2))
                )/u.Jy, y - mid_height/u.Jy])
            return z
        
        root1 = fsolve(gauss_function_intersection, [function.mean.value-1*u.um, function.mean.value+1*u.um])[0]
        return (function.mean.value - root1) * 2

def extract_data(file_name=str):
    raw_data = np.fromfile(os.path.abspath(file_name), sep=" ")
    return np.array(np.split(raw_data, len(raw_data)/2))

def loop_di_loop():
    y = 150
    for x in range(200, 300):
        data = (fits.open(os.path.abspath("cube_NII_Sh158_with_header.fits"))[0].data)
        spectrum = Spectrum(data[:,x,y])
        print(f"\n----------------\ncoords: {x,y}")
        spectrum.fit_NII()
        spectrum.plot_fit(fullscreen=True, coord=(x,y), plot_all=True)

loop_di_loop()

# data = (fits.open(os.path.abspath("cube_NII_Sh158_with_header.fits"))[0].data)
# spectrum = Spectrum(data[:,153,150])

# spectrum = Spectrum(extract_data(file_name="ds9.dat"))
# spectrum.fit_NII()
# spectrum.plot_fit()
