from astropy.io import fits
from astropy.wcs import WCS

from fits_analyzer import Data_cube, Map

import matplotlib.pyplot as plt
import numpy as np

"""
In this file are examples of code that have been used to create the fits files. Every operation has been grouped into a function.
"""

def get_smoothed_instr_f():
    """
    In this example, the smooth instrumental function map is calculated
    """
    # calibration_cube = 




def get_region_widening_maps(fwhm_map=Map, fwhm_unc_map=Map):
    """
    In this example, the four headers are extracted and attributed to the fwhm_map and the fwhm_unc_map. The data is also
    stored in gaussian_fitting/maps/reproject.
    """
    global_header = Map(fits.open("gaussian_fitting/data_cubes/night_34_wcs.fits")[0]).header
    header_0 = global_header.copy()
    wcs_0 = WCS(header_0)
    wcs_0.sip = None
    wcs_0 = wcs_0.dropaxis(2)
    header_0 = wcs_0.to_header(relax=True)

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
        ("global_widening.fits", fwhm_map, header_0), ("global_widening_unc.fits", fwhm_unc_map, header_0),
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


def get_turbulence_map():
    """
    In this example, the turbulence map is obtained with the previously computed maps: all region widenings and their
    uncertainties, smoothed_instr_f and its uncertainty. Note that the region widenings maps are not opened directly
    but are used in the Map.align_regions() method.
    """
    global_FWHM_map = Map(fits.open("gaussian_fitting/maps/reproject/global_widening.fits")[0])
    global_FWHM_map_unc = Map(fits.open("gaussian_fitting/maps/reproject/global_widening_unc.fits")[0])
    instrumental_function = Map(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")[0]).bin_map(2)
    instrumental_function_unc = Map(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f_unc.fits")[0]).bin_map(2)
    temp_map = Map(fits.open("gaussian_fitting/maps/external_maps/temp_nii_8300_pouss_snrsig2_seuil_sec_test95_avec_seuil_plus_que_0point35_incertitude_moins_de_1000.fits")[0])
    temp_map_unc = Map(fits.open("gaussian_fitting/maps/external_maps/temp_nii_8300_pouss_snrsig2_seuil_sec_test95_avec_seuil_plus_que_0point35_incertitude_moins_de_1000.fits")[0])
    temperature_map = temp_map.get_thermal_FWHM().get_reprojection(global_FWHM_map)
    temperature_map_unc = temp_map_unc.get_thermal_FWHM().get_reprojection(global_FWHM_map_unc)
    # The aligned maps are the result of the subtraction of the instrumental_function map squared to the global map squared
    aligned_map, aligned_map_unc = (global_FWHM_map**2 - instrumental_function**2).align_regions(
                                    global_FWHM_map.get_power_uncertainty(global_FWHM_map_unc, 2) + 
                                    instrumental_function.get_power_uncertainty(instrumental_function_unc, 2))
    turbulence_map = aligned_map - temperature_map**2
    turbulence_map_unc = turbulence_map.get_power_uncertainty(
                         aligned_map_unc + temperature_map.get_power_uncertainty(temperature_map_unc, 2), 0.5)
    turbulence_map **= 0.5
    
    diff = aligned_map**0.5 - (global_FWHM_map**2 - instrumental_function**2)**0.5

    plt.imshow(diff, vmin=0, vmax=15, origin="lower")
    # plt.imshow((global_FWHM_map**2-instrumental_function**2)**0.5, vmin=0, vmax=40, origin="lower")
    # plt.imshow(aligned_map**0.5, vmin=0, vmax=40, origin="lower")
    # plt.imshow(turbulence_map.data, vmin=0, vmax=40, origin="lower")
    plt.show()


get_turbulence_map()
