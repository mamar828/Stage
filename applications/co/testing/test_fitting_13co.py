import numpy as np
import graphinglib as gl
from astropy.io import fits
import os
import time
from eztcolors import Colors as C

from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.cubes.cube import Cube


cube = Cube.load("data/Loop4_co/N1/Loop4N1_13co.fits")
cube.header["CRPIX1"] = 2**15 - 1
cube.header["CRPIX2"] = 2**15 - 1
cube.header["CRPIX3"] = 2**15 - 1
cube = cube[500:800,:,:]

for y, map_ in enumerate(cube):
    for x, spectrum in enumerate(map_):
        if not spectrum.isnan():
            fig = gl.Figure(figure_style="dim")
            fig.add_elements(spectrum.plot, gl.Text(10, 0, f"({x},{y})", "lime"))
            fig.show()
