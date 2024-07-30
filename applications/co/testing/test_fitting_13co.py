import numpy as np
import graphinglib as gl
from astropy.io import fits
import os
import time
from eztcolors import Colors as C

from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.cubes.cube_co import CubeCO


cube = CubeCO.load("data/Loop4_co/N1/13co/Loop4N1_13co.fits")[500:800,:,:]

for y, map_ in enumerate(cube):
    for x, spectrum in enumerate(map_):
        if not spectrum.isnan():
            # if x == 13 and y == 32: pass
            # else: continue
            fig = gl.Figure(figure_style="dim", title=f"({x+1},{y+1})")

            spectrum.setattrs({
                "PEAK_PROMINENCE" : 0.2,
                "PEAK_MINIMUM_HEIGHT_SIGMAS" : 4,
                "PEAK_MINIMUM_DISTANCE" : 6,
                "PEAK_WIDTH" : 3,
                "NOISE_CHANNELS" : slice(0,150),
                "INITIAL_GUESSES_BINNING" : 2,
                "MAX_RESIDUE_SIGMAS" : 100,
                "STDDEV_DETECTION_THRESHOLD" : 0.1,
                "INITIAL_GUESSES_MAXIMUM_GAUSSIAN_STDDEV" : 10,
                "INITIAL_GUESSES_MINIMUM_GAUSSIAN_STDDEV" : 1,
            })
            spectrum.fit()
            
            fig1 = gl.Figure()
            fig1.add_elements(spectrum.plot)
            fig2 = gl.Figure()

            if spectrum.individual_functions_plot:
                fig1.add_elements(spectrum.initial_guesses_plot, *spectrum.individual_functions_plot)
                # fig1.add_elements(spectrum.total_functions_plot)
                fig2.add_elements(spectrum.residue_plot)
                print(spectrum.fit_results)
            else:
                fig2.add_elements(gl.Text(0.5, 0.5, "None"))
            multi_fig = gl.MultiFigure(2, 1, title=f"$(x,y)=$({x+1},{y+1})",)
            fig1.add_elements(gl.Curve([0, len(spectrum.data)], [spectrum.y_threshold, spectrum.y_threshold],
                                    label="y threshold", color="black", line_width=1))
            multi_fig.add_figure(fig1, 0, 0, 1, 1)
            multi_fig.add_figure(fig2, 1, 0, 1, 1)
            multi_fig.set_rc_params({"font.size": 9})
            multi_fig.show()
