import numpy as np
import matplotlib.pyplot as plt

from src.hdu.cubes.cube_co import CubeCO
from src.spectrums.spectrum_co import SpectrumCO


if __name__ == "__main__":
    cube = CubeCO.load("data/external/loop_co/Loop4N1_FinalJS.fits")[500:800,:,:]
    fig, axs = plt.subplots(1)
    cube.fit().plot(axs)
    plt.show()
