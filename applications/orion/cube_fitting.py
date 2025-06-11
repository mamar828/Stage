import src.graphinglib as gl
import numpy as np
from astropy.modeling import models

from src.hdu.cubes.fittable_cube import FittableCube
from src.hdu.maps.map import Map
from src.hdu.tesseract import Tesseract
from src.coordinates.ds9_coords import DS9Coords


cube = FittableCube.load("data/orion/data_cubes/nii_1.fits")
if cube.header.get("CRPIX3") is None and cube.header.comments["CRVAL3"] == "Velocity ref. of the 1st channel in km/s":
    cube.header["CRPIX3"] = (1, "Reference pixel for the velocity (manual fix)")
cube = cube[:, 250:255, 250:255]
def gaussian_model(x: float, *args):
    return sum([models.Gaussian1D.evaluate(x, amplitude=args[i], mean=args[i+1], stddev=args[i+2])
                for i in range(0, len(args), 3)])
def voigt_model(x, * args):
    return sum([models.Voigt1D().evaluate(x, amplitude_L=args[i], x_0=args[i+1], fwhm_L=args[i+2], fwhm_G=args[i+3])
                for i in range(0, len(args), 4)])

# Fit the data
# ------------
guesses = cube.find_peaks_gaussian_estimates(prominence=2, voigt=True)
# guesses.save("data/orion/nii/guesses_4.fits")
# calibration_fits = cube.fit(voigt_model, guesses, number_of_parameters=4, maxfev=1000000)
# calibration_fits.save("data/orion/nii_1_fits_tests.fits")

# See initial guesses
# -------------------
# guesses = FittableCube.load("data/orion/nii/guesses_4.fits")
for i in range(cube.data.shape[1]):
    for j in range(cube.data.shape[2]):
# for i in range(130, 350):
#     for j in range(130, 350):
        coords = i, j
        spec = cube[:, i, j]
        guess = guesses[:, i, j].data.reshape(-1, 4)
        voigt_guesses = [gl.Curve(spec.x_values, voigt_model(spec.x_values, *g)) for g in guess]
        gl.SmartFigure(elements=[cube[:,*coords].plot, *voigt_guesses], show_legend=False).show()

# See results
# -----------
# for i in range(cube.data.shape[1]):
#     for j in range(cube.data.shape[2]):
#         coords = i, j #DS9Coords(2, 1)
#         fits = Tesseract.load("data/orion/nii_1_fits_tests.fits")
#         print(fits.data[:,:,*coords])
#         plot = fits.get_spectrum_plot(cube, coords, voigt_model)
#         gl.SmartFigure(elements=[*plot]).show()

# Calculate FWHM
# --------------
# calibration_fits = Tesseract.load("data/orion/calibration_fits.fits")
# fit_maps = calibration_fits.to_grouped_maps(["amplitude_L", "x_0", "fwhm_L", "fwhm_G"])
# fwhm_L, fwhm_G = fit_maps.fwhm_L[0], fit_maps.fwhm_G[0]
# fwhm = 0.5343 * fwhm_L + (0.2169 * fwhm_L**2 + fwhm_G**2)**0.5
# fwhm.save("data/orion/calibration_fwhm.fits")

# Early testing
# -------------
# for i in range(cube.data.shape[1]):
#     for j in range(cube.data.shape[2]):
#         spec = cube[:, i, j]
#         # spec.plot.to_desmos(to_clipboard=True)
#         guess = guesses[:, i, j].data.reshape(-1, 4).flatten()
#         x_values = np.linspace(1, len(spec), 500)
#         # y_pred = np.zeros_like(x_values)
#         # for g in guess:
#         #     y_pred += models.Gaussian1D(g[0], g[1], g[2])(x_values) if not np.isnan(g[0]) else 0

#         spec_plot = spec.plot
#         spec_plot.line_width = 4
#         fit = gl.FitFromFunction(voigt_model,
#                                  spec.plot, label="Voigt Fit", guesses=guess, color="orange")
#         # print(i+1, j+1, fit.parameters)
#         gl.SmartFigure(figure_style="dark", elements=[
#             spec_plot,
#             gl.Curve.from_function(lambda x: voigt_model(x, *guess), x_min=1, x_max=48),
#             # gl.Curve(x_values, y_pred, label="Gaussian Estimates"),
#             fit,
#         ]).show()
