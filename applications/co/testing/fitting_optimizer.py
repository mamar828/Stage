import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from src.hdu.cubes.cube_co import CubeCO
from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.tesseract import Tesseract
from src.hdu.maps.grouped_maps import GroupedMaps


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
