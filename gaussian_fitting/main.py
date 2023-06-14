from astropy.io import fits
from astropy.wcs import WCS

from fits_analyzer import Data_cube, Map

import matplotlib.pyplot as plt
import numpy as np

"""
In this file are examples of code that have been used to create the fits files. Every operation has been grouped into a function.
"""

# def get_smoothed_instr_f():
#     calibration_cube = 



def get_region_widening_maps(fwhm_map=Map, fwhm_unc_map=Map):
    """
    In this example, the four headers are extracted and attributed to the fwhm_map and the fwhm_unc_map. The data is also
    stored in gaussian_fitting/maps/reproject.
    """
    global_header = Map(fits.open("gaussian_fitting/data_cubes/night_34_wcs.fits")[0]).header

    # Creation of a header per targeted region
    header_1 = global_header.copy()
    header_1["CRPIX1"] = 589
    header_1["CRPIX2"] = 477
    header_1["CRVAL1"] = (36.7706 + 13 * 60 + 23 * 3600)/(24 * 3600) * 360
    header_1["CRVAL2"] = 61 + (30 * 60 + 39.141)/3600
    header_1["CDELT1"] = -0.0005168263088 / 2.1458
    header_1["CDELT2"] = 0.0002395454546 / 1.0115
    wcs_1 = WCS(header_1)
    wcs_1.sip = None
    wcs_1 = wcs_1.dropaxis(2)
    header_1 = wcs_1.to_header(relax=True)

    header_2 = global_header.copy()
    header_2["CRPIX1"] = 642
    header_2["CRPIX2"] = 442
    header_2["CRVAL1"] = (30.2434 + 13 * 60 + 23 * 3600)/(24 * 3600) * 360
    header_2["CRVAL2"] = 61 + (30 * 60 + 10.199)/3600
    header_2["CDELT1"] = -0.0005168263088 / 2.1956
    header_2["CDELT2"] = 0.0002395454546 / 0.968
    wcs_2 = WCS(header_2)
    wcs_2.sip = None
    wcs_2 = wcs_2.dropaxis(2)
    header_2 = wcs_2.to_header(relax=True)

    header_3 = global_header.copy()
    header_3["CRPIX1"] = 674
    header_3["CRPIX2"] = 393
    header_3["CRVAL1"] = (26.2704 + 13 * 60 + 23 * 3600)/(24 * 3600) * 360
    header_3["CRVAL2"] = 61 + (29 * 60 + 28.387)/3600
    header_3["CDELT1"] = -0.0005168263088 / 2.14
    header_3["CDELT2"] = 0.0002395454546 / 0.942
    wcs_3 = WCS(header_3)
    wcs_3.sip = None
    wcs_3 = wcs_3.dropaxis(2)
    header_3 = wcs_3.to_header(relax=True)

    # Creation of every map with the matching WCS
    maps_to_create = [
        ("global_widening.fits", fwhm_map, global_header), ("global_widening_unc.fits", fwhm_unc_map, global_header),
        ("region_1_widening.fits", fwhm_map, header_1), ("region_1_widening_unc.fits", fwhm_unc_map, header_1),
        ("region_2_widening.fits", fwhm_map, header_2), ("region_2_widening_unc.fits", fwhm_unc_map, header_2),
        ("region_3_widening.fits", fwhm_map, header_3), ("region_3_widening_unc.fits", fwhm_unc_map, header_3)
    ]

    for filename, map_data, header in maps_to_create:
        # The header needs to be binned becaused the FWHM map is 512x512 pixels
        header_copy = header.copy()
        header_copy["CDELT1"] *= 2
        header_copy["CDELT2"] *= 2
        header_copy["CRPIX1"] /= 2
        header_copy["CRPIX2"] /= 2
        Map(fits.PrimaryHDU(map_data, header_copy)).save_as_fits_file(f"gaussian_fitting/maps/reproject/{filename}")


# get_region_widening_maps(fits.open("gaussian_fitting/maps/computed_data/fwhm_NII.fits")[0].data,
#                          fits.open("gaussian_fitting/maps/computed_data/fwhm_NII_unc.fits")[0].data)

