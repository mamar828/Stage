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


        # ------------- SHITSHOW ------------- ↴

        # We research the max peak in the region between channels 1 and 25
        max_intensity_x = list(self.y_values[0:25]).index(max(self.y_values[0:25]))
        self.max_tuple = (float(self.x_values[max_intensity_x]), float(self.y_values[max_intensity_x]), max_intensity_x + 1)

        # We store the second peak's location
        self.get_peak_bounds()
        temp_y_values = copy(self.y_values)
        temp_y_values[max_intensity_x] = 0
        second_peak_max_x = list(temp_y_values[0:25]).index(max(temp_y_values[0:25])) 
        self.second_max_tuple =  (float(self.x_values[second_peak_max_x]), 
                                  float(temp_y_values[second_peak_max_x]), second_peak_max_x + 1)

        # We store the right peak's location
        right_peak_max_x = list(self.y_values[25:47]).index(max(self.y_values[25:47])) + 25
        self.right_peak_max_tuple =  (float(self.x_values[right_peak_max_x]), 
                                  float(self.y_values[right_peak_max_x]), right_peak_max_x + 1)
        
        
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
        plt.ylabel("intensity")
        # print("uncertainties:",self.get_uncertainties())
        # print(self.get_fitted_gaussian_parameters())
        # print("stddev:", self.get_stddev(self.get_subtracted_fit()))
        fig.text(0.4, 0.92, f"coords: {coords}")
        if fullscreen == True:    
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        plt.show()

    @models.custom_model
    def gauss_function(x, a=1., x0=1., sigma=1., h=100.):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))+h

    def fit_NII(self):
        # Initialize the Gaussians
        g_init_OH1 = self.gauss_function(a=10, x0=0, h=4)
        g_init_OH2 = self.gauss_function(a=4, x0=19, h=4, bounds={"x0": (18, 20)})
        g_init_OH3 = self.gauss_function(a=4, x0=38, h=4, bounds={"x0": (36, 41)})
        g_init_OH4 = self.gauss_function(a=8, x0=47, h=4)
        g_init_NII = self.gauss_function(a=10, x0=14, h=4, bounds={"x0": (13, 15)})
        g_init_Ha  = self.gauss_function(a=20, x0=43, h=4, bounds={"x0": (42, 44)})
        g_init_OH1.x0.max = 4
        g_init_OH4.x0.min = 47
                
        gaussian_addition_init = g_init_OH1 + g_init_OH2 + g_init_OH2 + g_init_OH3 + g_init_OH4 + g_init_NII + g_init_Ha

        self.fit_g = fitting.LevMarLSQFitter()
        
        right_peak_bound = 25
        self.fitted_gaussian = self.fit_g(gaussian_addition_init, self.x_values, self.y_values)
        # print(self.fitted_gaussian)

    def get_fitted_gaussian_parameters(self):
        return self.fitted_gaussian
    
    def get_uncertainties(self):
        cov_matrix = self.fit_g.fit_info["param_cov"]
        return np.sqrt(np.diag(cov_matrix))
    
    def get_stddev(self, array):
        return np.std(array)

    def plot_fit(self, coord, fullscreen=False):
        self.plot(coord, fullscreen, fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit())

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
    for x in range(185, 300):
        data = (fits.open(os.path.abspath("cube_NII_Sh158_with_header.fits"))[0].data)
        spectrum = Spectrum(data[:,x,y])
        print(f"\n----------------\ncoords: {x,y}")
        spectrum.fit_NII()
        spectrum.plot_fit(fullscreen=True, coord=(x,y))

loop_di_loop()

# data = (fits.open(os.path.abspath("cube_NII_Sh158_with_header.fits"))[0].data)
# spectrum = Spectrum(data[:,153,150])

# spectrum = Spectrum(extract_data(file_name="ds9.dat"))
# spectrum.fit_NII()
# spectrum.plot_fit()
