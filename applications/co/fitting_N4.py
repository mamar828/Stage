import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from src.hdu.cubes.cube_co import CubeCO
from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.tesseract import Tesseract
from src.hdu.maps.grouped_maps import GroupedMaps
from src.hdu.maps.convenient_funcs import get_FWHM


if __name__ == "__main__":
    N4 = CubeCO.load("data/Loop4_co/N4/Loop4N4_Conv_Med_FinalJS.fits")[500:850,:,:]
    # N4.header["COMMENT"] = "Loop4N4_Conv_Med_FinalJS was binned 2x2."
    N4.header["COMMENT"] = "Loop4N4_Conv_Med_FinalJS was sliced at channel 500; all values of mean must then be " \
                         + "added to 500 to account for this shift."
    # N4.save("data/Loop4_co/N4/Loop4N4_Conv_Med_FinalJS_bin2.fits")

    """
    # Fitting the cube
    spectrum_parameters = {
        "peak_prominence" : 0.3,
        "peak_minimum_distance" : 6,
        "peak_width" : 4,
        "initial_guesses_binning" : 2,
        "max_residue_sigmas" : 7
    }

    chi2, fit_results = N4.fit(spectrum_parameters)
    chi2.save("data/Loop4_co/N4/chi2.fits")
    fit_results.save("data/Loop4_co/N4/tesseract.fits")
    """

    # """
    # Splitting, slicing and merging the Tesseract(s)
    print(f"Targeted channel : {N4.header.get_frame(1000, 0)}")
    fit_results = Tesseract.load("data/Loop4_co/N4/tesseract.fits")

    total = fit_results.filter(slice(135, 195))

    # Compressing the Tesseract
    total = total.compress()
    total.save(f"data/Loop4_co/N4/object.fits")
    # """

    # Harvesting data
    total = Tesseract.load(f"data/Loop4_co/N4/object.fits")
    gm = total.to_grouped_maps()
    fwhms = [get_FWHM(stddev_map, N4) for stddev_map in gm.stddev]
    GroupedMaps([("FWHM", fwhms)]).save(f"data/Loop4_co/N4/object_FWHM.fits")
