import numpy as np
import graphinglib as gl
from collections import namedtuple

from src.hdu.cubes.cube_co import CubeCO
from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.tesseract import Tesseract
from src.hdu.maps.grouped_maps import GroupedMaps
from src.hdu.maps.convenient_funcs import get_FWHM
from src.coordinates.ds9_coords import DS9Coords


if __name__ == "__main__":
    p = CubeCO.load("data/Loop4_co/p/Loop4p_Conv_Med_FinalJS.fits")[500:850,:,:].bin((1,2,2))
    p.header["COMMENT"] = "Loop4p_Conv_Med_FinalJS was binned 2x2."
    p.header["COMMENT"] = "Loop4p_Conv_Med_FinalJS was sliced at channel 500; all values of mean must then be " \
                         + "added to 500 to account for this shift."
    # p.save("data/Loop4_co/p/Loop4p_Conv_Med_FinalJS_bin2.fits")

    """
    # Fitting the cube
    spectrum_parameters = {
        "peak_prominence" : 0.4,
        "peak_minimum_distance" : 6,
        "peak_width" : 2.5,
        "initial_guesses_binning" : 2,
        "max_residue_sigmas" : 5
    }

    chi2, fit_results = p.fit(spectrum_parameters)
    chi2.save("data/Loop4_co/p/chi2.fits")
    fit_results.save("data/Loop4_co/p/tesseract.fits")
    """

    """
    # Splitting, slicing and merging the Tesseract(s)
    print(f"Targeted channel : {p.header.get_frame(0000, 0)}")
    fit_results = Tesseract.load("data/Loop4_co/p/tesseract.fits")

    total = fit_results.filter(slice(130, 205))

    # Compressing the Tesseract
    total = total.compress()
    total.save(f"data/Loop4_co/p/object.fits")
    """

    # Harvesting data
    total = Tesseract.load(f"data/Loop4_co/p/object.fits")
    fig = gl.Figure(size=(10,7))
    fig.add_elements(*total.get_spectrum_plot(p, DS9Coords(7, 20)))
    fig.show()

    gm = total.to_grouped_maps()
    fwhms = [get_FWHM(stddev_map, p) for stddev_map in gm.stddev]
    GroupedMaps([("FWHM", fwhms)]).save(f"data/Loop4_co/p/object_FWHM.fits")
