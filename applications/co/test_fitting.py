import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
import os

from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.cubes.cube import Cube


c = Cube.load("data/Loop4_co/N1/Loop4N1_FinalJS.fits")[500:800,:,:]
# c = Cube.load("data/external/loop_co/Loop4N2_Conv_Med_FinalJS.fits")[500:800,:,:]
# c = Cube.load("data/external/loop_co/Loop4N4_Conv_Med_FinalJS.fits")[500:850,:,:]
# c = Cube.load("data/external/loop_co/Loop4p_Conv_Med_FinalJS.fits")[500:850,:,:]
# s = SpectrumCO(c.data[:,17,20], c.header)
# s.auto_plot()

# for i in range(0*40+0, c.data.shape[1] * c.data.shape[2]):

for i in range(512, c.data.shape[1] * c.data.shape[2]):
    x = i % c.data.shape[2]
    y = i // c.data.shape[2]
    was_valid = True
    s = SpectrumCO(c.data[:,y,x], c.header)#, peak_minimum_height_sigmas=5)
    if not np.isnan(s.data).all():
        s.fit()
        # if not s.fit_valid:
        #     s.PEAK_MINIMUM_HEIGHT_SIGMAS -= 1
        #     s.fit()
        #     was_valid = False

        print(s.fit_results)
        chi2 = s.get_fit_chi2()
        print(chi2)

        fig, axs = plt.subplots(2)
        s.plot_fit(ax=axs[0], plot_initial_guesses=True, plot_all=True)
        s.plot_residue(ax=axs[1])
        fig.suptitle(f"$i={i}$     $(x,y)=$({x+1},{y+1})")
        axs[0].plot([0, len(s.data)], [s.y_threshold, s.y_threshold], "m-")
        fig.text(x=0.02, y=0.03, s="FWHM (km/s) :")
        if s.fitted_function:
            for i in s.fit_results.index:
                fig.text(x=0.2+i*0.17, y=0.03, s=s.get_FWHM_speed(i).round(3))
        fig.text(x=0.2, y=0.9, s=was_valid)
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        fig.tight_layout()
        plt.show()
        # input()

