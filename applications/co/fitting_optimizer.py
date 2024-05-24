import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from src.hdu.cubes.cube_co import CubeCO
from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.tesseract import Tesseract
from src.hdu.maps.grouped_maps import GroupedMaps



# Verify map validity by comparing pixels with test_fitting.py
fig, ax = plt.subplots(1)
GroupedMaps.load("data/internal/loop_co/Loop4N1_fit.fits").mean[3].data.plot(ax)
plt.show()
raise
if __name__ == "__main__":
    cube = CubeCO.load("data/external/loop_co/Loop4N1_FinalJS.fits")[500:800,:,:]
    # fig, axs = plt.subplots(1)
    # (cube[0,:,:].data*0).plot(axs, alpha=0.5, show_cbar=False)
    cube.header["COMMENT"] = "Loop4N1_FinalJS was previously sliced at channel 500, all values of mean must then be " \
                           + "added to 500 to account for this shift."
    chi2, fit_results = cube.fit()
    chi2.save("data/internal/loop_co/chi2.fits")
    results = fit_results.to_grouped_maps()
    results.save("data/internal/loop_co/Loop4N1_fit.fits")
    # mesospheric_ray.amplitude[0].data.plot(axs)
    # plt.show()
    