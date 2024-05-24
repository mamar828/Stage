import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from src.hdu.cubes.cube_co import CubeCO
from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.tesseract import Tesseract
from src.hdu.maps.grouped_maps import GroupedMaps


if __name__ == "__main__":
    cube = CubeCO.load("data/Loop4_co/N1/Loop4N1_FinalJS.fits")[500:800,:,:]
    print(500 + cube.header.get_frame(-4000, axis=0))
    cube.header["COMMENT"] = "Loop4N1_FinalJS was previously sliced at channel 500, all values of mean must then be " \
                           + "added to 500 to account for this shift."
    chi2, fit_results = cube.fit()
    chi2.save("data/Loop4_co/N1/fit_chi2.fits")
    fit_results.save("data/Loop4_co/N1/fit_tesseract.fits")

    # ----------------------------------------------------------------
    # Make cube output a spectrum if two ints are provided
    # ----------------------------------------------------------------

    # results = fit_results.to_grouped_maps()
    # results.save("data/Loop4_co/Loop4N1_fit.fits")

    # results = GroupedMaps.load("data/Loop4_co/Loop4N1_fit.fits")
    # object_ray = results[195:230]
    # fig, axs = plt.subplots(1)
    # (cube[0,:,:].data*0).plot(axs, alpha=0.5, show_cbar=False)
    # object_ray.mean[0].data.plot(axs)
    # plt.show()
