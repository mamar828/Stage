import os

import matplotlib.pyplot as plt
import numpy as np

from astropy.modeling import models, fitting
from astropy.io import fits
from copy import copy

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
                axs[0].plot(x_plot, value(x_plot), "r-", label=name)
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
            
        axs[0].plot(self.x_values, self.y_values, "g-", label="ds9 spectrum")
        axs[0].legend(loc="upper left", fontsize="8")
        axs[1].legend(loc="upper left", fontsize="8")
        plt.xlabel("channels")
        axs[0].set_ylabel("intensity")
        axs[1].set_ylabel("intensity")
        # print("----------------------- uncertainties -----------------------\n",self.get_uncertainties())
        # print(self.get_fitted_gaussian_parameters())
        # print("stddev:", self.get_stddev(self.get_subtracted_fit()))
        fig.text(0.4, 0.92, f"coords: {coords}, stddev: {self.get_stddev(self.get_subtracted_fit())}")
        fig.text(0.1, 0.92, self.peaks)
        if fullscreen:    
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        plt.show()

    def plot_fit(self, coord, fullscreen=False, plot_all=False):
        if plot_all:
            g = self.fitted_gaussian
            oh1 = self.gauss_function(a=g.a_0.value, x0=g.x0_0.value, h=g.h_0.value, sigma=g.sigma_0.value)
            oh2 = self.gauss_function(a=g.a_1.value, x0=g.x0_1.value, h=g.h_1.value, sigma=g.sigma_1.value)
            oh3 = self.gauss_function(a=g.a_2.value, x0=g.x0_2.value, h=g.h_2.value, sigma=g.sigma_2.value)
            oh4 = self.gauss_function(a=g.a_3.value, x0=g.x0_3.value, h=g.h_3.value, sigma=g.sigma_3.value)
            nii = self.gauss_function(a=g.a_4.value, x0=g.x0_4.value, h=g.h_4.value, sigma=g.sigma_4.value)
            ha  = self.gauss_function(a=g.a_5.value, x0=g.x0_5.value, h=g.h_5.value, sigma=g.sigma_5.value)
            self.plot(coord, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit(),
                      OH1=oh1, OH2=oh2, OH3=oh3, OH4=oh4, NII=nii, Ha=ha)
        
        else:
            self.plot(coord, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit())

    @models.custom_model
    def gauss_function(x, a=1., x0=1., sigma=2., h=0.):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))+h

    def fit_NII(self):
        # Initialize the Gaussians
        g_init_OH1 = self.gauss_function(a=10, x0=0)
        g_init_OH2 = self.gauss_function(a=3, x0=19, bounds={"x0": (18, 21)})
        g_init_OH3 = self.gauss_function(a=4, x0=38, bounds={"x0": (36, 39)})
        g_init_OH4 = self.gauss_function(a=8, x0=47)
        g_init_NII = self.gauss_function(a=10, x0=14, bounds={"x0": (13, 15)})
        g_init_Ha  = self.gauss_function(a=20, x0=43, bounds={"x0": (42, 44)})
        g_init_OH1.x0.max = 4
        g_init_OH4.x0.min = 47

        gaussian_addition_init = g_init_OH1 + g_init_OH2 + g_init_OH3 + g_init_OH4 + g_init_NII + g_init_Ha
        self.fit_g = fitting.LMLSQFitter(calc_uncertainties=True)
        self.fitted_gaussian = self.fit_g(gaussian_addition_init, self.x_values, self.y_values)
        self.get_initial_guesses()

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
                        print(derivatives_diff[x_list_deriv])
                    
                    if ray == "OH3":
                        if derivatives_diff[x_list_deriv] > diff_threshold_OH3:
                            x_peak = x
                            print(x_peak)
                            break

                    else:
                        if derivatives_diff[x_list_deriv] < diff_threshold and (
                            self.y_values[x_list] > self.y_values[x_peak-1] or x_peak == 0):
                            x_peak = x

            if x_peak == 0:
                # print(list(self.y_values[bounds[0]-1:bounds[1]-1]))
                x_peak = list(self.y_values[bounds[0]-1:bounds[1]-1]).index(max(self.y_values[bounds[0]-1:bounds[1]-1])) + bounds[0]
            
            x_peaks[ray] = x_peak
            
        self.peaks = x_peaks
        
        x_peak_OH1 = list(self.y_values[0:4]).index(max(self.y_values[0:4])) + 1        
        x_peak_OH2 = list(self.y_values[17:20]).index(max(self.y_values[17:20])) + 1 + 17
        x_peak_OH3 = list(self.y_values[36:41]).index(max(self.y_values[36:41])) + 1 + 36
        x_peak_OH4 = 48
        x_peak_NII = list(self.y_values[12:15]).index(max(self.y_values[12:15])) + 1 + 12
        x_peak_Ha  = list(self.y_values[41:44]).index(max(self.y_values[41:44])) + 1 + 41
        


    def get_fitted_gaussian_parameters(self):
        return self.fitted_gaussian
    
    def get_uncertainties(self):
        cov_matrix = self.fit_g.fit_info["param_cov"]
        return np.sqrt(np.diag(cov_matrix))
    
    def get_stddev(self, array):
        return np.std(array)

    def get_peak_bounds(self):
        # Determines the ratio of derivatives that identify a peak's boundaries
        sensitivity = 7

        # We do not consider the first seven channels for peaks
        derivatives = np.zeros(shape=(47,2))
        for i in range(0, len(self.x_values)-1):
            derivatives[i,0] = i + 1
            derivatives[i,1] = self.y_values[i+1] - self.y_values[i]
        
        lower_bound = 1
        higher_bound = 25


        for i in range(1, self.max_tuple[2]-1):
            if derivatives[i,1] / derivatives[i-1,1] > sensitivity or (derivatives[i,1] > 0 and derivatives[i-1,1] < 0):
                lower_bound = i + 1

        for i in range(self.max_tuple[2]+2, int(max(self.x_values))-1):
            if derivatives[i,1] / derivatives[i-1,1] < 1/sensitivity or (derivatives[i,1] > 0 and derivatives[i-1,1] < 0):
                higher_bound = i + 1
                break
        
        # The coordinates comprised in the bounds are [bounds[0]-1:bounds[1]
        self.bounds = lower_bound, higher_bound
        return lower_bound, higher_bound


    def get_peak_bound(self):
        # We'll look for the right peak in the region after channel 25
        derivatives = np.zeros(shape=(22,2))
        for i in range(25, len(self.x_values)-1):
            derivatives[i-25,0] = i + 1
            derivatives[i-25,1] = self.y_values[i+1] - self.y_values[i]

        lower_bound = 26

        # The closest permitted bound
        peak_closeness = 4
        
        for i in range(26, self.right_peak_max_tuple[2]-peak_closeness):
            if (derivatives[i-26,1] > 0 and derivatives[i-27,1] < 0):
                lower_bound = i

        self.lower_bound = lower_bound

        return lower_bound
        
    def get_subtracted_fit(self):
        subtracted_y = self.y_values - self.fitted_gaussian(self.x_values)
        return subtracted_y
    
    def get_FWHM(self, y_values):
        mid_height = (max(y_values) - self.fitted_gaussian.h.value)/2 + self.fitted_gaussian.h.value

        def function(xy):
            x, y = xy
            z = np.array([y - (self.fitted_gaussian.a.value*np.exp(
                -(x-self.fitted_gaussian.x0.value)**2/(2*self.fitted_gaussian.sigma.value**2))
                +self.fitted_gaussian.h.value), y - mid_height])
            return z
        
        root1 = fsolve(function, [self.max_tuple[2]-1, self.max_tuple[2]+1])[0]
        return (self.fitted_gaussian.x0.value - root1) * 2


def extract_data(file_name=str):
    raw_data = np.fromfile(os.path.abspath(file_name), sep=" ")
    return np.array(np.split(raw_data, len(raw_data)/2))

def loop_di_loop():
    y = 150
    for x in range(160, 300):
        data = (fits.open(os.path.abspath("cube_NII_Sh158_with_header.fits"))[0].data)
        spectrum = Spectrum(data[:,x,y])
        print(f"\n----------------\ncoords: {x,y}")
        spectrum.fit_NII()
        spectrum.plot_fit(fullscreen=True, coord=(x,y), plot_all=False)

loop_di_loop()

# data = (fits.open(os.path.abspath("cube_NII_Sh158_with_header.fits"))[0].data)
# spectrum = Spectrum(data[:,153,150])

# spectrum = Spectrum(extract_data(file_name="ds9.dat"))
# spectrum.fit_NII()
# spectrum.plot_fit()
