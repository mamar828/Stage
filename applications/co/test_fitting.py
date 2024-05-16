import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.cubes.cube import Cube


# c = Cube.load("data/external/loop_co/Loop4N1_FinalJS.fits")[500:800,:,:]
# c = Cube.load("data/external/loop_co/Loop4N2_Conv_Med_FinalJS.fits")[500:800,:,:]
# c = Cube.load("data/external/loop_co/Loop4N4_Conv_Med_FinalJS.fits")[500:850,:,:]
c = Cube.load("data/external/loop_co/Loop4p_Conv_Med_FinalJS.fits")[500:850,:,:]
# s = SpectrumCO(c.data[:,17,20], c.header)
# s.auto_plot()

for i in range(451, c.data.shape[1] * c.data.shape[2]):
    s = SpectrumCO(c.data[:,i // c.data.shape[2],i % c.data.shape[2]], c.header)
    if not np.isnan(s.data).all():
        s.fit()

        fig, axs = plt.subplots(2)
        s.plot_fit(ax=axs[0], plot_initial_guesses=True, plot_all=True)
        s.plot_residue(ax=axs[1])
        fig.suptitle(f"$i={i}$     $(x,y)=$({i % c.data.shape[2]+1},{i // c.data.shape[2]+1})")
        # axs[0].plot([0, len(s.data)], [s.y_threshold, s.y_threshold], "m-")
        fig.text(x=0.02, y=0.03, s="FWHM (km/s) :")
        for i in s.fit_results.index:
            fig.text(x=0.2+i*0.17, y=0.03, s=s.get_FWHM_speed(i).round(3))
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show()

