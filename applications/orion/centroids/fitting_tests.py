import src.graphinglib as gl
import numpy as np
from astropy.modeling import models

from src.hdu.cubes.fittable_cube import FittableCube
from src.hdu.maps.map import Map
from src.hdu.tesseract import Tesseract
from src.coordinates.ds9_coords import DS9Coords


cube = FittableCube.load("data/orion/data_cubes/binned/sii_1_binned_3x3.fits")
def gaussian_model(x, *args):
    return sum([models.Gaussian1D.evaluate(x, amplitude=args[i], mean=args[i+1], stddev=args[i+2])
                for i in range(0, len(args), 3)])
def voigt_model(x, *args):
    return sum([models.Voigt1D().evaluate(x, amplitude_L=args[i], x_0=args[i+1], fwhm_L=args[i+2], fwhm_G=args[i+3])
                for i in range(0, len(args), 4)])

# Fit the data
# ------------
# guesses = cube.range_peak_estimation([slice(9, 14), slice(29, 37)], voigt=True)
# guesses.save("data/orion/guesses.fits")
# guesses = FittableCube.load("data/orion/guesses.fits")
# guesses = np.dstack((10 + np.argmax(cube.data[9:14,:,:], axis=0), 29 + np.argmax(cube.data[29:34,:,:], axis=0)))

# param_bounds = [0, 1, 0, 0], [np.inf, int(cube.header["NAXIS3"]), 10, 10]
# fits = cube.fit(voigt_model, guesses, number_of_parameters=4, maxfev=10000,
#                 bounds=(guesses.shape[0]//4 * param_bounds[0], guesses.shape[0]//4 * param_bounds[1]))
# fits.save("data/orion/nii/nii_bounded.fits")


# See initial guesses
# -------------------
# guesses = FittableCube.load("data/orion/guesses.fits")
# guesses_num = gl.SmartFigure(elements=[gl.Heatmap(np.count_nonzero(~np.isnan(guesses.data), axis=0)/4,
#                                                   origin_position="lower")], aspect_ratio="equal")
# guesses_num[0][0].set_color_bar_params(label="Number of Guesses [-]")
# guesses_num.show()

# coords = DS9Coords(102, 124)
# spec = cube[:, *coords]
# guess = guesses[:, *coords].data.reshape(-1, 4)
# voigt_guesses = [gl.Curve(spec.x_values, voigt_model(spec.x_values, *g)) for g in guess]
# gl.SmartFigure(elements=[cube[:,*coords].plot, *voigt_guesses], show_legend=False).show()

# See results
# -----------
# for i in range(200, 300):
#     for j in range(200, 300):
# coords = i, j
coords = DS9Coords(102, 124)
fits = Tesseract.load("data/orion/fits/sii_1.fits")
print(fits.data[:,:,*coords])
# print(guesses.data[:,*coords])
plot = fits.get_spectrum_plot(cube, coords, voigt_model)
gl.SmartFigure(elements=[*plot]).show()

# Calculate FWHM
# --------------
# calibration_fits = Tesseract.load("data/orion/calibration_fits.fits")
# fit_maps = calibration_fits.to_grouped_maps(["amplitude_L", "x_0", "fwhm_L", "fwhm_G"])
# fwhm_L, fwhm_G = fit_maps.fwhm_L[0], fit_maps.fwhm_G[0]
# fwhm = 0.5343 * fwhm_L + (0.2169 * fwhm_L**2 + fwhm_G**2)**0.5
# fwhm.save("data/orion/calibration_fwhm.fits")
