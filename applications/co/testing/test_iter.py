import numpy as np
import matplotlib.pyplot as plt

from src.hdu.cubes.cube import Cube
from src.spectrums.spectrum_co import SpectrumCO


c = Cube.load("data/Loop4_co/N1/Loop4N1_FinalJS.fits")[500:800,:,:]
# c.bin((1,3,1)).save("t.fits")

spec = c[:,25,25].upgrade(SpectrumCO)
spec.fit()
fig, axs = plt.subplots(1)
spec.plot_fit(axs)
print(spec.get_FWHM_speed(0))
plt.show()

raise
for i, map_ in enumerate(c):
    # fig, axs = plt.subplots(1)
    # a = map_.data.plot(axs)
    # plt.show()
    for j, spec in enumerate(map_):
        if np.all(np.isnan(spec.data)):
            continue
        print(i,j)
        spec = spec.upgrade(SpectrumCO)
        spec.fit()
        print(spec.get_FWHM_speed(0))
        fig, axs = plt.subplots(1)
        spec.plot_fit(axs)
        plt.show()
