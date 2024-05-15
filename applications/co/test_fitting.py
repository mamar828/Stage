import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.cubes.cube import Cube


c = Cube.load("data/loop_co/Loop4N1_FinalJS.fits")[540:790,:,:]

# for x in range(c.data.shape[2]):
#     for y in range(c.data.shape[1]):
#         s = SpectrumCO(c.data[:,y,x], c.header)
for i in range(1107, c.data.shape[1] * c.data.shape[2]):
    s = SpectrumCO(c.data[:,i // c.data.shape[2],i % c.data.shape[2]], c.header)
    if not np.isnan(s.data).all():
        s.fit()
        print(s.fitted_function)

        fig, axs = plt.subplots(2)
        s.plot_fit(ax=axs[0], plot_initial_guesses=True, plot_all=True)
        s.plot_residue(ax=axs[1])
        fig.suptitle(f"i={i}")
        axs[0].plot([0, len(s.data)], [s.y_threshold, s.y_threshold], "m-")
        plt.show()


# Test bin algorithm

