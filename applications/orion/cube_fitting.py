import src.graphinglib as gl
import numpy as np
from astropy.modeling import models

from src.hdu.cubes.fittable_cube import FittableCube
from src.hdu.maps.map import Map
from src.hdu.tesseract import Tesseract
from src.coordinates.ds9_coords import DS9Coords


# cube = FittableCube.load("data/orion/data_cubes/nii_1.fits")
# if cube.header.get("CRPIX3") is None and cube.header.comments["CRVAL3"] == "Velocity ref. of the 1st channel in km/s":
#     cube.header["CRPIX3"] = (1, "Reference pixel for the velocity (manual fix)")
def gaussian_model(x, *args):
    return sum([models.Gaussian1D.evaluate(x, amplitude=args[i], mean=args[i+1], stddev=args[i+2])
                for i in range(0, len(args), 3)])
def voigt_model(x, *args):
    return sum([models.Voigt1D().evaluate(x, amplitude_L=args[i], x_0=args[i+1], fwhm_L=args[i+2], fwhm_G=args[i+3])
                for i in range(0, len(args), 4)])

# TESTER
# ------
x_vals = np.arange(10) + 1
y_vals = np.array([
    [
        gaussian_model(x_vals, 10, 4, 0.5) + gaussian_model(x_vals, 7, 8, 0.2),
        gaussian_model(x_vals, 9, 4.4, 0.6) + gaussian_model(x_vals, 6.7, 8.1, 0.3),
    ],
    [
        gaussian_model(x_vals, 11, 3.4, 0.55) + gaussian_model(x_vals, 7.1, 7.95, 0.1),
        gaussian_model(x_vals, 9.5, 3.8, 0.4) + gaussian_model(x_vals, 6, 7.3, 0.14),
    ],
]).T

cube = FittableCube(y_vals)
guesses = cube.range_peak_estimation([slice(2, 4), slice(6, 10)])

spectrums = [gl.Curve(x_vals, y) for y in FittableCube.flatten_3d_array(y_vals)]
gaussian_guesses = [gl.Curve.from_function(lambda x: gaussian_model(x, *guess), 1, 10)
                    for guess in FittableCube.flatten_3d_array(guesses.data)]

gl.SmartFigure(2, 2, elements=np.column_stack((spectrums, gaussian_guesses))).show()


# Fit the data
# ------------
# guesses = cube.find_peaks_gaussian_estimates(prominence=2, height=3, distance=15, voigt=True)
# guesses.save("data/orion/nii/guesses.fits")
# guesses = FittableCube.load("data/orion/nii/guesses.fits")#[:, 250:300, 50:100]
# guesses = np.dstack((10 + np.argmax(cube.data[9:14,:,:], axis=0), 29 + np.argmax(cube.data[29:34,:,:], axis=0)))

# param_bounds = [0, 1, 0, 0], [np.inf, int(cube.header["NAXIS3"]), 10, 10]
# fits = cube.fit(voigt_model, guesses, number_of_parameters=4, maxfev=10000,
#                 bounds=(guesses.shape[0]//4 * param_bounds[0], guesses.shape[0]//4 * param_bounds[1]))
# fits.save("data/orion/nii/nii_bounded.fits")


# See initial guesses
# -------------------
# guesses = FittableCube.load("data/orion/nii/guesses.fits")
# guesses_num = gl.SmartFigure(elements=[gl.Heatmap(np.count_nonzero(~np.isnan(guesses.data), axis=0)/4,
#                                                   origin_position="lower")], aspect_ratio="equal")
# guesses_num[0][0].set_color_bar_params(label="Number of Guesses [-]")
# guesses_num.show()

# coords = DS9Coords(334, 366)
# spec = cube[:, *coords]
# guess = guesses[:, *coords].data.reshape(-1, 4)
# voigt_guesses = [gl.Curve(spec.x_values, voigt_model(spec.x_values, *g)) for g in guess]
# gl.SmartFigure(elements=[cube[:,*coords].plot, *voigt_guesses], show_legend=False).show()

# See results
# -----------
# for i in range(200, 300):
#     for j in range(200, 300):
# coords = i, j
# coords = DS9Coords(136, 457)
# fits = Tesseract.load("data/orion/nii/nii_complete_new_estimates.fits")
# print(fits.data[:,:,*coords])
# # print(guesses.data[:,*coords])
# plot = fits.get_spectrum_plot(cube, coords, voigt_model)
# gl.SmartFigure(elements=[*plot]).show()

# Calculate FWHM
# --------------
# calibration_fits = Tesseract.load("data/orion/calibration_fits.fits")
# fit_maps = calibration_fits.to_grouped_maps(["amplitude_L", "x_0", "fwhm_L", "fwhm_G"])
# fwhm_L, fwhm_G = fit_maps.fwhm_L[0], fit_maps.fwhm_G[0]
# fwhm = 0.5343 * fwhm_L + (0.2169 * fwhm_L**2 + fwhm_G**2)**0.5
# fwhm.save("data/orion/calibration_fwhm.fits")
