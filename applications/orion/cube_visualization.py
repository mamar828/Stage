import src.graphinglib as gl
import numpy as np
from time import time
from astropy.modeling import models

from src.hdu.cubes.cube import Cube
from src.tools.messaging import notify_function_end


cube = Cube.load("data/orion/calibration.fits")[:, 240:260, 240:260]

guesses = cube.find_peaks_gaussian_estimates(prominence=10, distance=50)

for j in range(cube.data.shape[1]):
    for i in range(cube.data.shape[2]):
        spec = cube[:, j, i]
        # spec.plot.to_desmos(to_clipboard=True)
        guess = guesses[:, j, i].data.reshape(-1, 3)
        x_values = np.linspace(1, len(spec), 500)
        y_pred = np.zeros_like(x_values)
        for g in guess:
            y_pred += models.Gaussian1D(g[0], g[1], g[2])(x_values) if not np.isnan(g[0]) else 0
        
        spec_plot = spec.plot
        spec_plot.line_width = 4
        gl.SmartFigure(elements=[
            spec_plot,
            gl.Curve(x_values, y_pred, label="Gaussian Estimates"),
            # gl.FitFromFunction(models.Gaussian1D.evaluate, spec.plot, label="Gaussian Fit", guesses=g,
            #                    color=gl.get_color(color_number=1), line_style="-"),
            # gl.FitFromFunction(models.Lorentz1D.evaluate, spec.plot, label="Lorentzian Fit", guesses=g,
            #                    color=gl.get_color(color_number=2), line_style="-."),
            # gl.FitFromFunction(lambda x, x_0, A, fwhm_L, fwhm_G: models.Voigt1D().evaluate(x, x_0, A, fwhm_L, fwhm_G), 
            #                    spec.plot, label="Voigt Fit", guesses=[g[1], g[0], g[2], g[2]],
            #                    color=gl.get_color(color_number=3), line_style=":"),
        ]).show()
