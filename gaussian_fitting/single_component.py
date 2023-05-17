import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting


class gaussian_function:

    def __init__(self, compression, horizontal_translation):
        self.compression = compression
        self.horizontal_translation = horizontal_translation
        self.x_values = np.array([x for x in range(101)])
        self.y_values = np.e ** (- (compression * (self.x_values - horizontal_translation)) ** 2)

    def plot(self, randomized=False, other_list=None):
        if other_list is not None:
            plt.plot(self.x_values, other_list(self.x_values), "g-", label="fitted gaussian")
        if randomized:
            plt.plot(self.x_values, self.y_values_randomized, "r-", label="randomized gaussian")
        else:
            plt.plot(self.x_values, self.y_values, "b-", label="perfect gaussian")
        plt.legend(loc="upper left")
        plt.xlabel("channels")
        plt.ylabel("intensity")
        plt.show()

    def get_numpy_array(self):
        return np.stack((self.x_values, self.y_values), axis=1)
    
    def randomize(self, randomization_value):
        # randomization_value must be under one for best results
        randomized_addition = np.array(list((np.random.random() - 0.5) * randomization_value
                                       for y in range(len(self.y_values))))
        self.y_values_randomized = self.y_values + randomized_addition

# gaussian = gaussian_function(0.1, 50)
# gaussian.randomize(0.5)
# gaussian.plot(randomized=True)


class gaussian_fit(gaussian_function):

    def fit(self):
        x_max = self.x_values[list(self.y_values_randomized).index(max(self.y_values_randomized))]
        gaussian_parameters = models.Gaussian1D(amplitude=1., mean=x_max, stddev=1.)
        fit_gaussian = fitting.LevMarLSQFitter()        # Initialize the gaussian fitter
        self.fitted_gaussian = fit_gaussian(gaussian_parameters, self.x_values, self.y_values_randomized)

    def plot_fit(self):
        self.plot(other_list=self.fitted_gaussian, randomized=True)
        print(self.fitted_gaussian)

gaussian = gaussian_fit(0.1, 50)
gaussian.randomize(0.1)
gaussian.fit()
gaussian.plot_fit()



# # Generate fake data
# rng = np.random.default_rng(0)
# x = np.linspace(-5., 5., 200)
# y = 3 * np.exp(-0.5 * (x - 1.3)**2 / 0.8**2)
# y += rng.normal(0., 0.2, x.shape)

# # Fit the data using a Gaussian
# g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
# fit_g = fitting.LevMarLSQFitter()
# g = fit_g(g_init, x, y)

# # Plot the data with the best-fit model
# plt.figure(figsize=(8,5))
# plt.plot(x, y, 'ko')
# plt.plot(x, g(x), label='Gaussian')
# plt.xlabel('Position')
# plt.ylabel('Flux')
# plt.legend(loc=2)
# plt.show()
