import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from src.hdu.cubes.cube_co import CubeCO
from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.tesseract import Tesseract
from src.hdu.maps.grouped_maps import GroupedMaps
from src.hdu.maps.convenient_funcs import get_FWHM


if __name__ == "__main__":
    N1 = CubeCO.load("data/Loop4_co/N1/Loop4N1_FinalJS.fits")[500:800,:,:].bin((1,2,2))
    N1.header["COMMENT"] = "Loop4N1_FinalJS was binned 2x2."
    N1.header["COMMENT"] = "Loop4N1_FinalJS was sliced at channel 500, all values of mean must then be " \
                         + "added to 500 to account for this shift."
    # N1.save("data/Loop4_co/N1/Loop4N1_FinalJS_bin2.fits")

    """
    # Fitting the cube
    spectrum_parameters = {
        "peak_prominence" : 0.3,
        "peak_minimum_distance" : 6,
        "peak_width" : 2,
        "initial_guesses_binning" : 2
    }

    chi2, fit_results = N1.fit(spectrum_parameters)
    chi2.save("data/Loop4_co/N1/chi2.fits")
    fit_results.save("data/Loop4_co/N1/tesseract.fits")
    """

    """ 
    # Splitting, slicing and merging the Tesseract(s)
    print(f"Targeted channel : {N1.header.get_frame(-4000, 0)}")
    fit_results = Tesseract.load("data/Loop4_co/N1/tesseract.fits")

    splits = ["lower_left", "lower_right", "upper"]
    tesseract_splits = namedtuple("tesseract_splits", splits)
    lower, upper = fit_results.split(14, 2)
    lower_left, lower_right = lower.split(10, 3)
    tesseract_splits = tesseract_splits(lower_left, lower_right, upper)

    for split in splits:
        getattr(tesseract_splits, split).save(f"data/Loop4_co/N1/tesseract_splits/{split}.fits")

    upper = tesseract_splits.upper[200:]
    lower_left = tesseract_splits.lower_left[197:]
    lower_right = tesseract_splits.lower_right[190:]

    lower = lower_left.concatenate(lower_right, 3)
    total = lower.concatenate(upper, 2)

    # Compressing the Tesseract
    total = total.compress()
    total.save(f"data/Loop4_co/N1/object_compressed.fits")
    """

    # Harvesting data
    total = Tesseract.load(f"data/Loop4_co/N1/object_compressed.fits")
    gm = total.to_grouped_maps()
    fwhm = get_FWHM(gm.stddev[1], N1)
    fig, axs = plt.subplots(1)
    fwhm.data.plot(axs)
    plt.show()
