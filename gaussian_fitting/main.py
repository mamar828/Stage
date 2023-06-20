from astropy.io import fits
from astropy.wcs import WCS

from fits_analyzer import Data_cube, Map

import matplotlib.pyplot as plt
import numpy as np
import scipy

"""
In this file are examples of code that have been used to create the .fits files. Every operation has been grouped into
a function to improve readability.
"""

def get_smoothed_instr_f():
    """
    In this example, the smooth instrumental function map is calculated from the calibration_cube.
    """
    calibration_cube = Data_cube(fits.open("gaussian_fitting/data_cubes/calibration.fits")[0])
    calibration_map, calibration_map_unc = calibration_cube.fit_calibration()
    # For smoothing the change of interference order, a center pixel is required
    calibration_center_pixel = calibration_cube.get_center_point(center_guess=(527,484))
    calibration_center_pixel_rounded = round(calibration_center_pixel[0][0]), round(calibration_center_pixel[1][0])
    smoothed_instr_f, smoothed_instr_f_unc = calibration_map.smooth_order_change(
                                            calibration_map_unc, center=calibration_center_pixel_rounded)
    smoothed_instr_f.save_as_fits_file("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")
    smoothed_instr_f_unc.save_as_fits_file("gaussian_fitting/maps/computed_data/smoothed_instr_f_unc.fits")


# if __name__ == "__main__":
#     get_smoothed_instr_f()


def get_FWHM_maps():
    """
    In this example, the FWHM_NII maps are obtained.
    """
    nii_cube = Data_cube(fits.open("gaussian_fitting/data_cubes/night_34_wcs.fits")[0])
    # The 4 int indicates from which gaussian the FWHM will be extracted, in this case from the NII peak
    nii_map, nii_map_unc = nii_cube.bin_cube(2).fit(4)
    nii_map.save_as_fits_file("gaussian_fitting/maps/computed_data/fwhm_NII.fits")
    nii_map_unc.save_as_fits_file("gaussian_fitting/maps/computed_data/fwhm_NII_unc.fits")


# if __name__ == "__main__":
#     get_FWHM_maps()


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
        # The header needs to be binned becaused the FWHM map is 512x512 pixels and the header was made for 1024x1024 pixels
        header_copy = header.copy()
        header_copy["CDELT1"] *= 2
        header_copy["CDELT2"] *= 2
        header_copy["CRPIX1"] /= 2
        header_copy["CRPIX2"] /= 2
        Map(fits.PrimaryHDU(map_data, header_copy)).save_as_fits_file(f"gaussian_fitting/maps/reproject/{filename}")


# get_region_widening_maps(fits.open("gaussian_fitting/maps/computed_data/fwhm_NII.fits")[0].data,
#                          fits.open("gaussian_fitting/maps/computed_data/fwhm_NII_unc.fits")[0].data)


def get_turbulence_map(temp_map, temp_map_unc):
    """
    In this example, the turbulence map is obtained with the previously computed maps: all region widenings and their
    uncertainties as well as smoothed_instr_f and its uncertainty. Note that the region widenings maps are not opened
    directly but are used in the Map.align_regions() method.
    """
    global_FWHM_map = Map(fits.open("gaussian_fitting/maps/reproject/global_widening.fits")[0])
    global_FWHM_map_unc = Map(fits.open("gaussian_fitting/maps/reproject/global_widening_unc.fits")[0])
    instrumental_function = Map(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")[0]).bin_map(2)
    instrumental_function_unc = Map(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f_unc.fits")[0]).bin_map(2)
    # The temperature maps are adjusted at the same WCS than the global maps
    temperature_map = temp_map.transfer_temperature_to_FWHM().reproject_on(global_FWHM_map)
    temperature_map_unc = temp_map_unc.transfer_temperature_to_FWHM().reproject_on(global_FWHM_map_unc)
    # The aligned maps are the result of the subtraction of the instrumental_function map squared to the global map squared
    aligned_map, aligned_map_unc = (global_FWHM_map**2 - instrumental_function**2).align_regions(
                                    global_FWHM_map.calc_power_uncertainty(global_FWHM_map_unc, 2) + 
                                    instrumental_function.calc_power_uncertainty(instrumental_function_unc, 2))
    turbulence_map = aligned_map - temperature_map**2
    turbulence_map_unc = turbulence_map.calc_power_uncertainty(
                         aligned_map_unc + temperature_map.calc_power_uncertainty(temperature_map_unc, 2), 0.5)
    turbulence_map **= 0.5
    # The standard deviation is the desired quantity
    turbulence_map /= 2 * np.sqrt(2 * np.log(2))
    turbulence_map_unc /= 2 * np.sqrt(2 * np.log(2))
    turbulence_map.save_as_fits_file("gaussian_fitting/maps/computed_data/turbulence.fits")
    turbulence_map_unc.save_as_fits_file("gaussian_fitting/maps/computed_data/turbulence_unc.fits")


# get_turbulence_map(Map(fits.open("gaussian_fitting/maps/external_maps/temp_it_nii_8300.fits")[0]),
#                    Map(fits.open("gaussian_fitting/maps/external_maps/temp_it_nii_err_8300.fits")[0]))


def get_temperature_from_NII_and_SII():
    """
    In this example, we obtain a temperature map using Courtes's method with the NII and SII emission lines.
    """
    # The SII broadening map is acquired
    sii_FWHM = Map(fits.open("gaussian_fitting/leo/SII_FWHM+header.fits")[0])
    temp_in_fwhm = Map.transfer_temperature_to_FWHM(fits.PrimaryHDU(np.full((sii_FWHM.data.shape), 8500), None))
    sii_FWHM_with_temperature = (sii_FWHM**2 + temp_in_fwhm**2)**0.5
    sii_FWHM_with_temperature.data[sii_FWHM_with_temperature.data > 10000] = np.NAN
    sii_sigma_with_temperature = sii_FWHM_with_temperature / (2 * np.sqrt(2 * np.log(2)))

    # The NII turbulence map is acquired
    nii_sigma = Map(fits.open("gaussian_fitting/maps/computed_data/turbulence.fits")[0])
    temp_map_FWHM = Map(fits.open(
        "gaussian_fitting/maps/external_maps/temp_it_nii_8300.fits")[0]).transfer_temperature_to_FWHM().reproject_on(nii_sigma)
    nii_sigma_with_temperature = (nii_sigma**2 + (temp_map_FWHM / (2*np.sqrt(2*np.log(2))))**2)**0.5

    # The FWHM maps are converted in Angstroms
    sii_peak_AA = 6716
    nii_peak_AA = 6583.41
    
    sii_sigma_with_temperature = 1000 * sii_sigma_with_temperature * sii_peak_AA / scipy.constants.c
    nii_sigma_with_temperature = 1000 * nii_sigma_with_temperature * nii_peak_AA / scipy.constants.c

    # The two maps are used to compute a temperature map
    temperature_map = 4.15 * 10**4 * (nii_sigma_with_temperature**2 - 
                       sii_sigma_with_temperature.reproject_on(nii_sigma_with_temperature)**2)
    temperature_map.plot_map()


# get_temperature_from_NII_and_SII()


def get_turbulence_from_Halpha():
    """
    In this example, turbulence maps from the Halpha ray are obtained and saved.
    """
    cube = Data_cube(fits.open("gaussian_fitting/data_cubes/night_34_wcs.fits")[0])
    # The 4 int indicates from which gaussian the FWHM will be extracted, in this case from the NII peak
    # ha_map, ha_map_unc = cube.bin_cube(2).fit(4)
    # ha_map.save_as_fits_file("gaussian_fitting/maps/computed_data/halpha/fwhm_NII.fits")
    # ha_map_unc.save_as_fits_file("gaussian_fitting/maps/computed_data/halpha/fwhm_NII_unc.fits")



get_turbulence_from_Halpha()

