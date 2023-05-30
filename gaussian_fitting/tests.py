import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
import os
from astropy.io import fits


"""x = np.array([x for x in range(101)])
y = np.e ** (- (0.1 * (x - 50)) ** 2)
y += np.array(list((np.random.random() - 0.5) * 0.25 for y in range(len(y))))




# Generate fake data
# rng = np.random.default_rng(0)
# x = np.linspace(-5., 5., 200)
# y = 3 * np.exp(-0.5 * (x - 1.3)**2 / 0.8**2)
# y += rng.normal(0., 0.2, x.shape)

# Fit the data using a Gaussian
g_init = models.Gaussian1D(amplitude=1., mean=50, stddev=1.)
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init, x, y)

# Plot the data with the best-fit model
plt.figure(figsize=(8,5))
plt.plot(x, y, 'ko')
plt.plot(x, g(x), label='Gaussian')
plt.xlabel('Position')
plt.ylabel('Flux')
plt.legend(loc=2)
plt.show()"""



# ----------------------------------------------------------------------------------------------------------



# data = np.fromfile(os.path.abspath("test.dat"), sep=" ")
# print(data)
# real_data = np.array(np.split(data, 2))
# print(real_data)


# fits_data = (fits.open(os.path.abspath("calibration.fits"))[0].data)
# fits_wcs = WCS((fits.open('calibration.fits', mode = 'denywrite'))[0].header)
# plt.plot(fits_data[:,150,150])
# plt.show()



# ----------------------------------------------------------------------------------------------------------


"""
data = (fits.open(os.path.abspath("calibration.fits"))[0].data)
rdata = data[:,0,600]

x_values, y_values = np.arange(48) + 1, rdata

g_init = models.Gaussian1D(amplitude=100., mean=30, stddev=1.)
fit_g = fitting.LevMarLSQFitter()
bounds = (26,32)

fitted_gaussian = fit_g(g_init, x_values[bounds[0]-1:bounds[1]], y_values[bounds[0]-1:bounds[1]])
mean = (np.sum(y_values[0:bounds[0]-1]) + np.sum(y_values[bounds[1]+1:48])) / (
    max(x_values) - (bounds[1] - bounds[0] + 1))



plt.plot(x_values, fitted_gaussian(x_values) + mean, label="fitted")
plt.plot(x_values, y_values, "y--", label="ds9 spectrum")
plt.legend(loc="upper left")
plt.xlabel("channels")
plt.ylabel("intensity")
plt.show()
"""



# --------------------------------------------------------------------


# a = True
# b = True
# c = False
# print(a == b and (a == a))


# -------------------------------------------------------------

# g_init = models.Voigt1D(x_0=self.max_tuple[0], amplitude_L=500, fwhm_L=3., fwhm_G=3.5)
        # g_init = models.Gaussian1D(amplitude=1., mean=self.max_tuple[0], stddev=1.)
        # g_init = gauss_function(self.x_values, a=500., x0=self.max_tuple, sigma=1., h=100.)
        # fit_g = fitting.LevMarLSQFitter()
        # bounds = self.get_peak_bounds()
        # self.fitted_gaussian = fit_g(g_init, self.x_values[bounds[0]-1:bounds[1]], self.y_values[bounds[0]-1:bounds[1]])



# -----------------------------------------------------------------------------

# stack = np.stack((x_values, self.fitted_gaussian(x_values)), axis=1)
# b = []
# for i in stack:
#     b.append(tuple(i))

# print(b)




# -------------------------------------------------------------------------------------------

# a = np.zeros(shape=(1,2))
# print(a)


# -------------------------------------------------------------------------------------------------------

# a = " [1.55726605e+01 2.88057290e+01 9.92512489e+00 4.79773984e+07"
# n = 0
# for i in a:
#     n += 1
# print(n)


# -----------------------------------------------------------------------------

# def func(a, **other):
#     print(a)
#     for e, i in other.items():
#         print(i)

# func("lol", lol=["binou", "toupie"])



# -----------------------------------------------------------------------------------


# numbers = [2, 3, 7, 4, 8]

# sum(number**2 for number in numbers if number % 2 == 0)
# exec("result = sum(number**2 for number in numbers if number % 2 == 0)")
# print(result)

# pro = "allo = 3"
# exec(pro)
# print(allo)




# -----------------------------------------

# a = int()

# print(a == 0)



# ---------------------------------------------

# fits_data = (fits.open('calibration.fits'))[0].data

# plt.plot(fits_data[:,0,0])
# plt.show()

# ------------------------------------------------------

# binou = None
# binou = 3
# if binou:
#     print(binou)

# ----------------------------------------------------------

# file = open("writer.txt", "a")
# file.write("SCHLOUBABOOBAfasddsfa\n")

# file = open("writer.txt", "r")
# print(file.read())


# --------------------------------------------------------------------

nuit_3 = fits.open("lambda_3.fits")[0].data
nuit_4 = fits.open("lambda_4.fits")[0].data
header = fits.open("lambda_3.fits")[0].header

nuit_34 = np.flip(np.sum((nuit_3, nuit_4), axis=0), axis=(1,2))
plt.imshow(nuit_34[15,:,:])
plt.show()
fits.writeto("night_34.fits", nuit_34, header, overwrite=True)
print("d")
