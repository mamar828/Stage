import src.graphinglib as gl
import numpy as np
from astropy.modeling import models

from src.hdu.cubes.fittable_cube import FittableCube
from src.hdu.maps.map import Map
from src.hdu.tesseract import Tesseract
from src.coordinates.ds9_coords import DS9Coords


# Bin and fix the cube
# --------------------
# calib = FittableCube.load("data/orion/calibration/calibration_cube.fits")
# calib.header.rename_keyword("RADECSYS", "RADESYSa")
# calib.header.remove("VELREF")
# calib = calib.bin([4, 1, 1])
# calib.save("data/orion/calibration/calibration_binned.fits")

# cube = FittableCube.load("data/orion/calibration/calibration_binned.fits")
# def gaussian_model(x, *args):
#     return sum([models.Gaussian1D.evaluate(x, amplitude=args[i], mean=args[i+1], stddev=args[i+2])
#                 for i in range(0, len(args), 3)])
# def voigt_model(x: float, A: float, x_0: float, fwhm_L: float, fwhm_G: float) -> float:
#     return (
#         models.Voigt1D().evaluate(x, amplitude_L=A, x_0=x_0, fwhm_L=fwhm_L, fwhm_G=fwhm_G)
#         + models.Voigt1D().evaluate(x-cube.shape[0], amplitude_L=A, x_0=x_0, fwhm_L=fwhm_L, fwhm_G=fwhm_G)
#     )

# Fit the data
# ------------
# guesses = cube.find_peaks_estimation(voigt=True, prominence=1, distance=200)
# calibration_fits = cube.fit(voigt_model, guesses, number_of_parameters=4)
# calibration_fits.save("data/orion/calibration/voigt_fits.fits")

# Centroid maps
# -------------
# tess = Tesseract.load("data/orion/calibration/voigt_fits.fits")
# centroids = tess.to_grouped_maps(["amplitude_L", "x_0", "fwhm_L", "fwhm_G"]).x_0
# assert len(centroids) == 1, "Expected only one component in the fits."
# centroids[0].save("data/orion/calibration/calibration_centroids.fits")

# See results
# -----------
# coords = DS9Coords(174, 219)
# calibration_fits = Tesseract.load("data/orion/calibration/calibration_fits_gaussian.fits")
# plot = calibration_fits.get_spectrum_plot(cube, coords, gaussian_model)
# calibration_fits = Tesseract.load("data/orion/calibration/calibration_fits.fits")
# plot2 = calibration_fits.get_spectrum_plot(cube, coords, voigt_model)
# plot[1].label = "Gaussian Fit"
# plot2[1].label = "Voigt Fit"
# plot[1].line_width = 1
# plot2[1].line_width = 1
# fig = gl.SmartFigure(elements=[*plot[:2], plot2[1]], size=(20,16),
#                      x_label="Channel Number [-]", y_label="Intensity [arb. u.]")
# fig.set_ticks(x_tick_spacing=10, y_tick_spacing=50, minor_x_tick_spacing=1, minor_y_tick_spacing=10)
# fig.set_grid()
# fig.save("figures/orion/calibration/voigt_vs_gaussian.pdf")

# Calculate FWHM
# --------------
# calibration_fits = Tesseract.load("data/orion/calibration/calibration_fits_gaussian.fits")
# fit_maps = calibration_fits.to_grouped_maps(["amplitude", "mean", "stddev"])

# fwhm = 2 * np.sqrt(2 * np.log(2)) * fit_maps.stddev[0]
# fwhm.save("data/orion/calibration/calibration_gaussian_fwhm.fits")

# Early testing
# -------------
# for j in range(cube.data.shape[1]):
#     for i in range(cube.data.shape[2]):
#         spec = cube[:, j, i]
#         # spec.plot.to_desmos(to_clipboard=True)
#         guess = guesses[:, j, i].data.reshape(-1, 4)
#         x_values = np.linspace(1, len(spec), 500)
#         y_pred = np.zeros_like(x_values)
#         for g in guess:
#             y_pred += models.Gaussian1D(g[0], g[1], g[2])(x_values) if not np.isnan(g[0]) else 0

#         spec_plot = spec.plot
#         spec_plot.line_width = 4
#         fit = gl.FitFromFunction(voigt_model,
#                                  spec.plot, label="Voigt Fit", guesses=g, color="orange")
#         print(i+1, j+1, fit.parameters)
#         gl.SmartFigure(figure_style="dark", elements=[
#             spec_plot,
#             # gl.Curve(x_values, y_pred, label="Gaussian Estimates"),
#             fit,
#         ]).show()

# Gaussian vs Voigt
# -----------------
# gaussian_fwhm = Map.load("data/orion/calibration/calibration_gaussian_fwhm.fits")
# voigt_fwhm = Map.load("data/orion/calibration/calibration_fwhm.fits")
# (abs(gaussian_fwhm - voigt_fwhm) / voigt_fwhm).save("data/orion/calibration/calibration_fwhm_relative_difference.fits")
