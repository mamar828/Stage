import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import time
from eztcolors import Colors as C

from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.cubes.cube import Cube


# c = Cube.load("data/Loop4_co/N1/Loop4N1_FinalJS.fits")[500:800,:,:].bin((1,2,2))
c = Cube.load("data/Loop4_co/N2/Loop4N2_Conv_Med_FinalJS.fits")[500:800,:,:]
# c = Cube.load("data/Loop4_co/N4/Loop4N4_Conv_Med_FinalJS.fits")[500:850,:,:]
# c = Cube.load("data/Loop4_co/p/Loop4p_Conv_Med_FinalJS.fits")[500:850,:,:].bin((1,2,2))
# c.save("t.fits")

# for i in range(0*40+0, c.data.shape[1] * c.data.shape[2]):
for i in range(517, c.data.shape[1] * c.data.shape[2]):
    x = i % c.data.shape[2]
    y = i // c.data.shape[2]
    """ Parameters for N1 """
    # s = SpectrumCO(c.data[:,y,x], c.header, peak_prominence=0.3, peak_minimum_distance=6, peak_width=2,
    #                initial_guesses_binning=2)
    """ Parameters for N2 """
    s = SpectrumCO(c.data[:,y,x], c.header, peak_prominence=0.0, peak_minimum_distance=6, peak_width=2,
                   initial_guesses_binning=2)
    """ Parameters for N4 """
    # s = SpectrumCO(c.data[:,y,x], c.header, peak_prominence=0.3, peak_minimum_distance=6, peak_width=4,
    #                initial_guesses_binning=2, max_residue_sigmas=7)
    """ Parameters for p """
    # s = SpectrumCO(c.data[:,y,x], c.header, peak_prominence=0.4, peak_minimum_distance=6, peak_width=2.5,
    #                initial_guesses_binning=2, max_residue_sigmas=5)
    if not np.isnan(s.data).all():
        s.fit()

        print(s.fit_results)
        fig, axs = plt.subplots(2)
        s.plot_fit(ax=axs[0], plot_initial_guesses=True, plot_all=True)
        s.plot_residue(ax=axs[1])
        fig.suptitle(f"$i={i}$     $(x,y)=$({x+1},{y+1})")
        axs[0].plot([0, len(s.data)], [s.y_threshold, s.y_threshold], "m-")
        fig.text(x=0.02, y=0.03, s="FWHM (km/s) :")
        if s.fitted_function:
            for j in s.fit_results.index:
                fig.text(x=0.2+j*0.17, y=0.03, s=s.get_FWHM_speed(j).round(3))
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        fig.tight_layout()
        plt.show()

        while not s.is_well_fitted and s.fitted_function is not None:
            print(f"{C.LIGHT_RED}REFITTING{C.END}")
            # input(i)
            new = s.copy()
            new.fit()
            if new.get_fit_chi2() < s.get_fit_chi2():
                s = new
            else:
                break
            print(s.fit_results)

            fig, axs = plt.subplots(2)
            rect = fig.patch
            rect.set_facecolor("blue") 
            s.plot_fit(ax=axs[0], plot_initial_guesses=True, plot_all=True)
            s.plot_residue(ax=axs[1])
            fig.suptitle(f"$i={i}$     $(x,y)=$({x+1},{y+1})")
            axs[0].plot([0, len(s.data)], [s.y_threshold, s.y_threshold], "m-")
            fig.text(x=0.02, y=0.03, s="FWHM (km/s) :")
            if s.fitted_function:
                for j in s.fit_results.index:
                    fig.text(x=0.2+j*0.17, y=0.03, s=s.get_FWHM_speed(j).round(3))
            # manager = plt.get_current_fig_manager()
            # manager.full_screen_toggle()
            fig.tight_layout()
            plt.show()
            # input()

