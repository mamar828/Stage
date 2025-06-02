import src.graphinglib as gl
import numpy as np
from astropy.modeling import models

from src.hdu.cubes.fittable_cube import FittableCube
from src.hdu.maps.map import Map
from src.hdu.tesseract import Tesseract
from src.coordinates.ds9_coords import DS9Coords


cube = FittableCube.load("data/orion/data_cubes/calibration.fits")
def voigt_model(x: float, A: float, x_0: float, fwhm_L: float, fwhm_G: float) -> float:
    return (
        models.Voigt1D().evaluate(x, amplitude_L=A, x_0=x_0, fwhm_L=fwhm_L, fwhm_G=fwhm_G)
        + models.Voigt1D().evaluate(x-cube.shape[0], amplitude_L=A, x_0=x_0, fwhm_L=fwhm_L, fwhm_G=fwhm_G)
    )

# Fit the data
# ------------
# guesses = cube.find_peaks_gaussian_estimates(voigt=True, prominence=1, distance=200)
# calibration_fits = cube.fit(voigt_model, guesses, number_of_tasks=500)
# calibration_fits.save("data/orion/calibration_fits_3.fits")

# See results
# -----------
# coords = DS9Coords(118, 78)
# calibration_fits = Tesseract.load("data/orion/calibration_fits.fits")
# print(calibration_fits.data[:,:,*coords])
# plot = calibration_fits.get_spectrum_plot(cube, coords, voigt_model)
# gl.SmartFigure(elements=[*plot]).show()

# Calculate FWHM
# --------------
# calibration_fits = Tesseract.load("data/orion/calibration_fits.fits")
# fit_maps = calibration_fits.to_grouped_maps(["amplitude_L", "x_0", "fwhm_L", "fwhm_G"])
# fwhm_L, fwhm_G = fit_maps.fwhm_L[0], fit_maps.fwhm_G[0]
# fwhm = 0.5343 * fwhm_L + (0.2169 * fwhm_L**2 + fwhm_G**2)**0.5
# fwhm.save("data/orion/calibration_fwhm.fits")

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
