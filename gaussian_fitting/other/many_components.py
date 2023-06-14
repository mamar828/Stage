import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
import os


class Spectrum:

    def __init__(self, data=np.ndarray):
        self.data = data
        self.x_values, self.y_values = np.split(data, 2, axis=1)
        # Detect beforehand the maximum intensity to establish a scale 
        max_position = list(self.y_values).index(max(self.y_values))
        self.max_tuple = (float(self.x_values[max_position]), float(self.y_values[max_position]), max_position)
    
    def plot(self, **other_values):
        for name, value in other_values.items():
            try:
                plt.plot(self.x_values, value(self.x_values), label=name)
            except:
                try:
                    plt.plot(self.x_values, value, label=name)
                except:
                    plt.plot(value[:,0], value[:,1], "og", label=name)
        plt.plot(self.x_values, self.y_values, "y--", label="ds9 spectrum")
        plt.legend(loc="upper left")
        plt.xlabel("channels")
        plt.ylabel("intensity")
        plt.show()

    def fit(self, mean):
        g_init = models.Gaussian1D(amplitude=1., mean=mean, stddev=1.)
        fit_g = fitting.LevMarLSQFitter()        # Initialize the gaussian fitter
        return fit_g(g_init, self.x_values[38:48], self.y_values[38:48])

    def fit_single(self):
        gaussian_parameters = models.Gaussian1D(amplitude=1., mean=self.max_tuple[0], stddev=1.)
        fit_gaussian = fitting.LevMarLSQFitter()        # Initialize the gaussian fitter
        self.fitted_gaussian = fit_gaussian(gaussian_parameters, self.x_values[38:48], self.y_values[38:48])

    def plot_fit(self):
        self.plot(test=self.fitted_gaussian)

    def detect_emission_rays(self, sensitivity=float):
        """
        Detects the emission peaks and their widths.

        Arguments:
        ---------
        Sensitivity: between 0 and 1, specifies the precision of the detection of peaks. A greater sensitivity
        will result in more peaks that might not all be real peaks. A smaller sensitivity may not be able to
        find all peaks.

        Returns:
        -------
        List of tuples representing each peak: [(mean, width), ...]
        """

        # peak_cutoff = sensitivity * self.max_tuple[2]

        # y_differences = [(float(self.y_values[i+1] - self.y_values[i]), i) for i in range(len(self.y_values) - 1)]
        # # print(y_differences)

        # peak_candidates = []

        # for y_difference, position in y_differences:
        #     if y_difference > peak_cutoff:
        #         peak_candidates.append((y_difference, position))
        
        # print(peak_candidates)

        derivatives = np.zeros(shape=(47,1))
        x_value_step = self.x_values[1] - self.x_values[0]

        for x in range(len(self.x_values)-1):
            derivatives[x] = (self.y_values[x+1] - self.y_values[x]) / x_value_step
        
        peak_candidates = np.zeros(shape=(47,2))

        for x in range(1, len(derivatives)):
            if derivatives[x-1] * derivatives[x] < 0:
                peak_candidates[x,1] = derivatives[x]
                peak_candidates[x,0] = x

        self.peak_candidates = peak_candidates[peak_candidates[:,0] != 0]





    
    def smooth_distribution(self):
        """
        Smooths out the distribution to ease the peaks' detection.
        """
        self.y_values_smoothed = np.zeros(shape=(48,1))
        self.y_values_smoothed[0] = (self.y_values[0] + self.y_values[1]) / 2
        for i in range(1, len(self.y_values) - 1):
            self.y_values_smoothed[i] = (self.y_values[i-1] + self.y_values[i] + self.y_values[i+1]) / 3
        self.y_values_smoothed[-1] = (self.y_values[-1] + self.y_values[-2]) / 2


raw_data = np.fromfile(os.path.abspath("ds9.dat"), sep=" ")
data = np.array(np.split(raw_data, len(raw_data)/2))

testing_pixel = Spectrum(data)
testing_pixel.fit_single()
# testing_pixel.smooth_distribution()
# testing_pixel.detect_emission_rays(0.10)
testing_pixel.plot()
# testing_pixel.plot_fit()





