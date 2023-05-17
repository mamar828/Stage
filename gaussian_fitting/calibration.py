import os

import matplotlib.pyplot as plt
import numpy as np

from astropy.modeling import models, fitting
from astropy.io import fits




def gauss_function(x, a, x0, sigma, h):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+h

class Spectrum:

    def __init__(self, data=np.ndarray, displacement=True, desired_peak_position=35):
        self.data = data
        self.displacement = displacement

        try:
            self.x_values, self.y_values = np.split(data, 2, axis=1)
        except Exception:
            self.x_values, self.y_values = np.arange(48) + 1, data

        max_intensity_x = list(self.y_values).index(max(self.y_values))
        self.max_tuple = (float(self.x_values[max_intensity_x]), float(self.y_values[max_intensity_x]), max_intensity_x + 1)

        if not displacement:
            peak_position_translation = desired_peak_position - self.max_tuple[2]
            new_y_values = np.zeros(shape=48)

            for i, value in enumerate(self.y_values):
                if (peak_position_translation + i) >= 48:
                    new_y_values[i+peak_position_translation-48] = value
                else:
                    new_y_values[i+peak_position_translation] = value

            self.old_y_values = self.y_values
            self.y_values = new_y_values
            max_intensity_x = list(self.y_values).index(max(self.y_values))
            self.max_tuple = (float(self.x_values[max_intensity_x]), float(self.y_values[max_intensity_x]), max_intensity_x + 1)
    
    def plot(self, **other_values):
        for name, value in other_values.items():
            try:
                x_plot = np.arange(1,49,0.05)
                plt.plot(x_plot, value(x_plot), "r-", label=name)
            except:
                try:
                    plt.plot(self.x_values, value, label=name)
                except:
                    plt.plot(value[:,0], value[:,1], "og", label=name)
        plt.plot(self.x_values, self.y_values_modified, "y--", label="translated spectrum")
        plt.plot(self.x_values, self.old_y_values, "g:", label="ds9 spectrum")
        plt.legend(loc="upper left")
        plt.xlabel("channels")
        plt.ylabel("intensity")
        plt.show()

    def fit_single(self):
        bounds = self.get_peak_bounds()
        # The coordinates comprised in the bounds are [bounds[0]-1:bounds[1]]
        print("bounds:", bounds)

        if self.displacement:
            self.mean = (np.sum(self.y_values[0:bounds[0]-1]) + np.sum(self.y_values[bounds[1]:48])) / (
                max(self.x_values) - (bounds[1] - bounds[0] + 1))
        else:
            self.mean = np.sum(self.y_values[0:25]) / 25

        self.y_values_modified = self.y_values - self.mean
        g_init = models.Gaussian1D(amplitude=1., mean=self.max_tuple[0], stddev=1.)
        fit_g = fitting.LevMarLSQFitter()
        self.fitted_gaussian = fit_g(g_init, self.x_values[bounds[0]-1:bounds[1]], self.y_values_modified[bounds[0]-1:bounds[1]])
        
        # g_init = models.Voigt1D(x_0=self.max_tuple[0], amplitude_L=500, fwhm_L=3., fwhm_G=3.5)
        # g_init = models.Gaussian1D(amplitude=1., mean=self.max_tuple[0], stddev=1.)
        # g_init = gauss_function(self.x_values, a=500., x0=self.max_tuple, sigma=1., h=100.)
        # fit_g = fitting.LevMarLSQFitter()
        # bounds = self.get_peak_bounds()
        # self.fitted_gaussian = fit_g(g_init, self.x_values[bounds[0]-1:bounds[1]], self.y_values[bounds[0]-1:bounds[1]])
        

    def plot_fit(self):
        self.plot(fit=self.fitted_gaussian)

    def get_peak_bounds(self):
        # Determines the ratio of derivatives that identify a peak's boundaries
        sensitivity = 7

        derivatives = np.zeros(shape=(47,2))
        for i in range(0, len(self.x_values)-1):
            derivatives[i,0] = i + 1
            derivatives[i,1] = self.y_values[i+1] - self.y_values[i]
        
        lower_bound = 1
        higher_bound = 48

        for i in range(1, self.max_tuple[2]-1):
            if derivatives[i,1] / derivatives[i-1,1] > sensitivity or (derivatives[i,1] > 0 and derivatives[i-1,1] < 0):
                lower_bound = i + 1

        for i in range(self.max_tuple[2]+2, int(max(self.x_values))-1):
            if derivatives[i,1] / derivatives[i-1,1] < 1/sensitivity or (derivatives[i,1] > 0 and derivatives[i-1,1] < 0):
                higher_bound = i + 1
                break
        
        return lower_bound, higher_bound

        # interval = min(self.max_tuple[2] - lower_bound, higher_bound - self.max_tuple[2])
        # return (self.max_tuple[2] - interval, self.max_tuple[2] + interval)

def extract_data(file_name=str):
    raw_data = np.fromfile(os.path.abspath(file_name), sep=" ")
    return np.array(np.split(raw_data, len(raw_data)/2))

def loop_di_loop():
    for x in range(300):
        data = (fits.open(os.path.abspath("calibration.fits"))[0].data)
        spectrum = Spectrum(data[:,x,200], displacement=False)
        print(f"\n-----------------\ncoords: {x,200}\n-----------------")
        spectrum.fit_single()
        spectrum.plot_fit()

loop_di_loop()

# data = (fits.open(os.path.abspath("calibration.fits"))[0].data)
# spectrum = Spectrum(data[:,600,600])

spectrum = Spectrum(extract_data(file_name="ds9.dat"))
spectrum.fit_single()
spectrum.plot_fit()




# fits_data = (fits.open(os.path.abspath("calibration.fits"))[0].data)
# fits_wcs = WCS((fits.open('calibration.fits', mode = 'denywrite'))[0].header)
# plt.plot(fits_data[:,150,150])
# plt.show()
