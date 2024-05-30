import numpy as np
import matplotlib.pyplot as plt

from src.hdu.cubes.cube_co import CubeCO
from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.tesseract import Tesseract
from src.hdu.maps.grouped_maps import GroupedMaps


if __name__ == "__main__":
    N1 = CubeCO.load("data/Loop4_co/N1/Loop4N1_FinalJS.fits")[500:800,:,:].bin((1,2,2))
    N2 = CubeCO.load("data/Loop4_co/N2/Loop4N2_Conv_Med_FinalJS.fits")[500:800,:,:]
    N4 = CubeCO.load("data/Loop4_co/N4/Loop4N4_Conv_Med_FinalJS.fits")[500:850,:,:]
    p = CubeCO.load("data/Loop4_co/p/Loop4p_Conv_Med_FinalJS.fits")[500:850,:,:].bin((1,2,2))

    """ Parameters for N1 """
    # s = SpectrumCO(c.data[:,y,x], c.header, peak_prominence=0.3, peak_minimum_distance=6, peak_width=2,
    #                initial_guesses_binning=2)
    """ Parameters for N2 """
    s = SpectrumCO(c.data[:,y,x], c.header, peak_prominence=0.0, peak_minimum_distance=6, peak_width=2.5,
                initial_guesses_binning=2, max_residue_sigmas=5)
    """ Parameters for N4 """
    # s = SpectrumCO(c.data[:,y,x], c.header, peak_prominence=0.3, peak_minimum_distance=6, peak_width=4,
    #                initial_guesses_binning=2, max_residue_sigmas=7)
    """ Parameters for p """
    # s = SpectrumCO(c.data[:,y,x], c.header, peak_prominence=0.4, peak_minimum_distance=6, peak_width=2.5,
    #                initial_guesses_binning=2, max_residue_sigmas=5)







if __name__ == "__main__":
    cube = CubeCO.load("data/Loop4_co/N1/Loop4N1_FinalJS.fits")[500:800,:,:]
    cube.header["COMMENT"] = "Loop4N1_FinalJS was previously sliced at channel 500, all values of mean must then be " \
                           + "added to 500 to account for this shift."
    spectrum_parameters = {
        "peak_prominence" : 0.7,
        "peak_minimum_height_sigmas" : 5.0,
        "peak_minimum_distance" : 10,
        "noise_channels" : slice(0,100)
    }
    chi2_list = []
    for i in np.linspace(2, 20, 40):
        print(i)
        spectrum_parameters["peak_minimum_distance"] = i
        chi2 = cube.fit(spectrum_parameters)[0]
        chi2_list.append([i, np.nanmean(chi2.data)])

    array = np.array(chi2_list)
    plt.plot(array[:,0], array[:,1])
    plt.xlabel("peak_minimum_distance")
    plt.ylabel("chi2")
    plt.savefig("figures/peak_minimum_distance.png", dpi=600, bbox_inches="tight")
