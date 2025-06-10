import numpy as np
import graphinglib as gl
from astropy.io import fits
import os
import time
from eztcolors import Colors as C

from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.cubes.cube import Cube


# c = Cube.load("data/Loop4/N1/Loop4N1_FinalJS.fits")[500:800,:,:].bin((1,2,2))
# c = Cube.load("data/Loop4/N2/Loop4N2_Conv_Med_FinalJS_wcs.fits")[500:800,:,:]
c = Cube.load("data/Loop4/N4/Loop4N4_Conv_Med_FinalJS_wcs.fits")[500:850,:,:]#[:,:,30:]
# c = Cube.load("data/Loop4/p/Loop4p_Conv_Med_FinalJS_wcs.fits")[500:850,:,:].bin((1,2,2))

for i in range(22*45+25, c.data.shape[1] * c.data.shape[2]):
# for i in range(1458, c.data.shape[1] * c.data.shape[2]):
    x = i % c.data.shape[2]
    y = i // c.data.shape[2]
    print(x, y)
    """ Parameters for N1 """
    # s = SpectrumCO(c.data[:,y,x], c.header, peak_prominence=0.3, peak_minimum_distance=6, peak_width=2,
    #                initial_guesses_binning=2, max_residue_sigmas=6, initial_guesses_maximum_gaussian_stddev=10)
    """ Parameters for N2 """
    # s = SpectrumCO(c.data[:,y,x], c.header, peak_prominence=0.2, peak_minimum_distance=6, peak_width=2,
    #                initial_guesses_binning=2, max_residue_sigmas=5, initial_guesses_maximum_gaussian_stddev=7)
    """ Parameters for N4 """
    s = SpectrumCO(c.data[:,y,x], c.header, peak_prominence=0.2, peak_minimum_distance=6, peak_width=2,
                   initial_guesses_binning=2, max_residue_sigmas=5, initial_guesses_maximum_gaussian_stddev=7)
    """ Parameters for p """
    # s = SpectrumCO(c.data[:,y,x], c.header, peak_prominence=0.4, peak_minimum_distance=6, peak_width=2.5,
    #                initial_guesses_binning=2, max_residue_sigmas=5, initial_guesses_maximum_gaussian_stddev=7)
    if not np.isnan(s.data).all():
        s.fit()

        print(s.fit_results)
        print(s.initial_guesses)
        a = False

        while not s.is_well_fitted and s.fitted_function is not None:
            # s.auto_plot()
            print(f"{C.LIGHT_RED}REFITTING{C.END}")
            # input(i)
            new = s.copy()
            new.fit()
            if new.get_fit_chi2() < s.get_fit_chi2():
                s = new
            else:
                break
            # if new.max_residue < s.max_residue:
            #     s = new
            # else:
            #     break
            print(s.fit_results)

        if a:
            input()
        fig1 = gl.Figure()
        fig1.add_elements(s.plot)
        fig2 = gl.Figure()
        if s.individual_functions_plot:
            fig1.add_elements(s.initial_guesses_plot, *s.individual_functions_plot)
            fig1.add_elements(s.total_functions_plot)
            fig2.add_elements(s.residue_plot)
        else:
            fig2.add_elements(gl.Text(0.5, 0.5, "None"))
        multi_fig = gl.MultiFigure(2, 1, title=f"$i={i}$     $(x,y)=$({x+1},{y+1})",)
        fig1.add_elements(gl.Curve([0, len(s.data)], [s.y_threshold, s.y_threshold],
                                   label="y threshold", color="black", line_width=1))
        multi_fig.add_figure(fig1, 0, 0, 1, 1)
        multi_fig.add_figure(fig2, 1, 0, 1, 1)
        multi_fig.set_rc_params({"font.size": 9})
        multi_fig.show()
