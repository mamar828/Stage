import src.graphinglib as gl
import numpy as np
from astropy.modeling import models

from src.hdu.cubes.cube import Cube
from src.hdu.maps.map import Map


cube = Cube.load("data/orion/data_cubes/calibration.fits")


guesses = cube.find_peaks_gaussian_estimates(voigt=True, prominence=10, distance=200)
voigt_model = lambda x, A, x_0, fwhm_L, fwhm_G: models.Voigt1D().evaluate(x, amplitude_L=A, x_0=x_0,
                                                                          fwhm_L=fwhm_L, fwhm_G=fwhm_G)
fits = cube.fit(voigt_model, guesses)#, number_of_tasks=100)

fwhm = Map(
    data=(0.5343 * fits.data[2, :, :] + np.sqrt(0.2169 * fits.data[2, :, :]**2 + fits.data[3, :, :]**2)),
    header=cube.header.flatten(0),
)
fwhm.save("data/orion/calibration_fwhm.fits")


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
