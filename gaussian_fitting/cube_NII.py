import os

import matplotlib.pyplot as plt
import numpy as np

from astropy.modeling import models, fitting
from astropy.io import fits    

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
    
    @models.custom_model
    def gauss_function(x, a=1., x0=1., sigma=1., h=100.):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))+h

    def plot(self, **other_values):
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
            
        axs[0].plot(self.x_values, self.y_values, "y--", label="translated spectrum")
        if self.displacement:
            axs[0].plot(self.x_values, self.y_values, "g:", label="ds9 spectrum")
        else:
            axs[0].plot(self.x_values, self.old_y_values, "g:", label="ds9 spectrum")
        axs[0].legend(loc="upper left", fontsize="8")
        axs[1].legend(loc="upper left", fontsize="8")
        plt.xlabel("channels")
        plt.ylabel("intensity")
        # print("uncertainties:",self.get_uncertainties())
        # print(self.get_fitted_gaussian_parameters())
        # print("stddev:", self.get_stddev(self.get_subtracted_fit()))
        plt.show()

    def fit_single(self):
        g_init = self.gauss_function(a=self.max_tuple[1], x0=self.max_tuple[2])
        self.fit_g = fitting.LevMarLSQFitter()
        
        self.fitted_gaussian = self.fit_g(g_init, self.x_values, self.y_values)

    def get_fitted_gaussian_parameters(self):
        return self.fitted_gaussian
    
    def get_uncertainties(self):
        cov_matrix = self.fit_g.fit_info["param_cov"]
        return np.sqrt(np.diag(cov_matrix))
    
    def get_stddev(self, array):
        return np.std(array)

    def plot_fit(self):
        self.plot(fit=self.fitted_gaussian, subtracted_fit=self.get_subtracted_fit())

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
        
        # The coordinates comprised in the bounds are [bounds[0]-1:bounds[1]]
        return lower_bound, higher_bound
    
    def get_subtracted_fit(self):
        subtracted_y = self.y_values - self.fitted_gaussian(self.x_values)
        return subtracted_y


def extract_data(file_name=str):
    raw_data = np.fromfile(os.path.abspath(file_name), sep=" ")
    return np.array(np.split(raw_data, len(raw_data)/2))

def loop_di_loop():
    for x in range(100, 300):
        data = (fits.open(os.path.abspath("cube_NII_Sh158_with_header.fits"))[0].data)
        spectrum = Spectrum(data[:,x,120], displacement=True)
        print(f"\n----------------\ncoords: {x,200}")
        spectrum.fit_single()
        spectrum.plot_fit()

# loop_di_loop()

data = (fits.open(os.path.abspath("cube_NII_Sh158_with_header.fits"))[0].data)
spectrum = Spectrum(data[:,153,150])

# spectrum = Spectrum(extract_data(file_name="ds9.dat"))
spectrum.fit_single()
spectrum.plot_fit()
