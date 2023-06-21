from astropy.io import fits
from astropy.wcs import WCS

from fits_analyzer import Data_cube, Map, Map_u

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
    calibration_map = calibration_cube.fit_calibration()
    # For smoothing the change of interference order, a center pixel is required
    calibration_center_pixel = calibration_cube.get_center_point(center_guess=(527,484))
    calibration_center_pixel_rounded = round(calibration_center_pixel[0][0]), round(calibration_center_pixel[1][0])
    smoothed_instr_f = calibration_map.smooth_order_change(center=calibration_center_pixel_rounded)
    smoothed_instr_f.save_as_fits_file("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")


# if __name__ == "__main__":
#     get_smoothed_instr_f()


def get_FWHM_maps():
    """
    In this example, the FWHM_NII maps are obtained.
    """
    nii_cube = Data_cube(fits.open("gaussian_fitting/data_cubes/night_34_wcs.fits")[0])
    # The 4 int indicates from which gaussian the FWHM will be extracted, in this case from the NII peak
    nii_map = nii_cube.bin_cube(2).fit(4)
    nii_map.save_as_fits_file("gaussian_fitting/maps/computed_data/fwhm_NII.fits")


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
        ("global_widening.fits", header_0),
        ("region_1_widening.fits", header_1),
        ("region_2_widening.fits", header_2),
        ("region_3_widening.fits", header_3)
    ]

    for filename, header in maps_to_create:
        # The header needs to be binned becaused the FWHM map is 512x512 pixels and the header was made for 1024x1024 pixels
        header_copy = header.copy()
        header_copy["CDELT1"] *= 2
        header_copy["CDELT2"] *= 2
        header_copy["CRPIX1"] /= 2
        header_copy["CRPIX2"] /= 2

        data_map = Map_u(fits.HDUList([fits.PrimaryHDU(fwhm_map, header_copy),
                                       fits.ImageHDU(fwhm_unc_map, header_copy)]))
        data_map.save_as_fits_file(f"gaussian_fitting/maps/reproject/--{filename}")


# get_region_widening_maps(fits.open("gaussian_fitting/maps/computed_data/fwhm_NII.fits")[0].data,
#                          fits.open("gaussian_fitting/maps/computed_data/fwhm_NII.fits")[1].data)



def get_turbulence_map(temp_map):
    """
    In this example, the turbulence map is obtained with the previously computed maps: all region widenings and their
    uncertainties as well as smoothed_instr_f and its uncertainty. Note that the region widenings maps are not opened
    directly but are used in the Map.align_regions() method.
    """
    global_FWHM_map = Map_u(fits.open("gaussian_fitting/maps/reproject/global_widening.fits"))
    instrumental_function = Map_u(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")).bin_map(2)
    # The temperature maps are adjusted at the same WCS than the global maps
    temperature_map = temp_map.transfer_temperature_to_FWHM().reproject_on(global_FWHM_map)
    # The aligned maps are the result of the subtraction of the instrumental_function map squared to the global map squared
    aligned_map = (global_FWHM_map**2 - instrumental_function**2).align_regions()
    turbulence_map = (aligned_map - temperature_map**2)**0.5
    # The standard deviation is the desired quantity
    turbulence_map /= 2 * np.sqrt(2 * np.log(2))
    turbulence_map.save_as_fits_file("gaussian_fitting/maps/computed_data/turbulence.fits")


# get_turbulence_map(Map_u(fits.HDUList([fits.open("gaussian_fitting/maps/external_maps/temp_it_nii_8300.fits")[0],
#                                        fits.open("gaussian_fitting/maps/external_maps/temp_it_nii_err_8300.fits")[0]])))


def get_courtes_temperature_from_NII_and_SII():
    """
    In this example, we obtain a temperature map using Courtes's method with the NII and SII emission lines.
    """
    # Here the SII emission line sigma is obtained (includes the temperature's contribution)
    # A global temperature of 8500K was used
    sii_FWHM = Map(fits.open("gaussian_fitting/leo/SII/SII_FWHM+header.fits")[0]) * 2*np.sqrt(2*np.log(2))
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
    
    sii_sigma_with_temperature_AA = 1000 * sii_sigma_with_temperature * sii_peak_AA / scipy.constants.c
    nii_sigma_with_temperature_AA = 1000 * nii_sigma_with_temperature * nii_peak_AA / scipy.constants.c

    # The two maps are used to compute a temperature map
    temperature_map = 4.73 * 10**4 * (nii_sigma_with_temperature_AA**2 - 
                       sii_sigma_with_temperature_AA.reproject_on(nii_sigma_with_temperature_AA)**2)
    temperature_map.save_as_fits_file("courtes_temperature_NII_SII.fits")
    # temperature_map.plot_map()


# get_courtes_temperature_from_NII_and_SII()


def get_courtes_temperature_from_NII_and_Halpha():
    """
    In this example, we obtain a temperature map using Courtes's method with the NII and Halpha emission lines.
    """
    # Here the Halpha emission line sigma is obtained (includes the temperature's contribution)
    # A global temperature of 8500K was used
    halpha_FWHM = Map(fits.open("gaussian_fitting/leo/Halpha/Halpha_FWHM+header.fits")[0]) * 2*np.sqrt(2*np.log(2))
    temp_in_fwhm = Map.transfer_temperature_to_FWHM(fits.PrimaryHDU(np.full((halpha_FWHM.data.shape), 8500), None))
    halpha_FWHM_with_temperature = (halpha_FWHM**2 + temp_in_fwhm**2)**0.5
    halpha_FWHM_with_temperature.data[halpha_FWHM_with_temperature.data > 10000] = np.NAN
    halpha_sigma_with_temperature = halpha_FWHM_with_temperature / (2 * np.sqrt(2 * np.log(2)))

    # The NII turbulence map is acquired
    nii_sigma = Map(fits.open("gaussian_fitting/maps/computed_data/turbulence.fits")[0])
    temp_map_FWHM = Map(fits.open(
        "gaussian_fitting/maps/external_maps/temp_it_nii_8300.fits")[0]).transfer_temperature_to_FWHM().reproject_on(nii_sigma)
    nii_sigma_with_temperature = (nii_sigma**2 + (temp_map_FWHM / (2*np.sqrt(2*np.log(2))))**2)**0.5

    # The FWHM maps are converted in Angstroms
    halpha_peak_AA = 6562.8
    nii_peak_AA = 6583.41
    
    halpha_sigma_with_temperature_AA = 1000 * halpha_sigma_with_temperature * halpha_peak_AA / scipy.constants.c
    nii_sigma_with_temperature_AA = 1000 * nii_sigma_with_temperature * nii_peak_AA / scipy.constants.c

    # The two maps are used to compute a temperature map
    temperature_map = 4.73 * 10**4 * (nii_sigma_with_temperature_AA**2 - 
                       halpha_sigma_with_temperature_AA.reproject_on(nii_sigma_with_temperature_AA)**2)
    temperature_map.plot_map()


# get_courtes_temperature_from_NII_and_Halpha()


def get_courtes_temperature_from_NII_and_OIII():
    """
    In this example, we obtain a temperature map using Courtes's method with the NII and OIII emission lines.
    """
    # Here the OIII emission line sigma is obtained (includes the temperature's contribution)
    # A global temperature of 8500K was used
    oiii_FWHM = Map(fits.open("gaussian_fitting/leo/OIII/OIII_FWHM+header.fits")[0]) * 2*np.sqrt(2*np.log(2))
    temp_in_fwhm = Map.transfer_temperature_to_FWHM(fits.PrimaryHDU(np.full((oiii_FWHM.data.shape), 8500), None))
    oiii_FWHM_with_temperature = (oiii_FWHM**2 + temp_in_fwhm**2)**0.5
    oiii_FWHM_with_temperature.data[oiii_FWHM_with_temperature.data > 10000] = np.NAN
    oiii_sigma_with_temperature = oiii_FWHM_with_temperature / (2 * np.sqrt(2 * np.log(2)))

    # The NII turbulence map is acquired
    nii_sigma = Map(fits.open("gaussian_fitting/maps/computed_data/turbulence.fits")[0])
    temp_map_FWHM = Map(fits.open(
        "gaussian_fitting/maps/external_maps/temp_it_nii_8300.fits")[0]).transfer_temperature_to_FWHM().reproject_on(nii_sigma)
    nii_sigma_with_temperature = (nii_sigma**2 + (temp_map_FWHM / (2*np.sqrt(2*np.log(2))))**2)**0.5

    # The FWHM maps are converted in Angstroms
    oiii_peak_AA = 5007
    nii_peak_AA = 6583.41
    
    oiii_sigma_with_temperature_AA = 1000 * oiii_sigma_with_temperature * oiii_peak_AA / scipy.constants.c
    nii_sigma_with_temperature_AA = 1000 * nii_sigma_with_temperature * nii_peak_AA / scipy.constants.c

    # The two maps are used to compute a temperature map
    temperature_map = 4.73 * 10**4 * (nii_sigma_with_temperature_AA**2 - 
                       oiii_sigma_with_temperature_AA.reproject_on(nii_sigma_with_temperature_AA)**2)
    temperature_map.plot_map()


get_courtes_temperature_from_NII_and_OIII()


def get_turbulence_from_Halpha():
    """
    In this example, turbulence maps from the Halpha ray are obtained and saved.
    """
    cube = Data_cube(fits.open("gaussian_fitting/data_cubes/night_34_wcs.fits")[0])
    # The 5 int indicates from which gaussian the FWHM will be extracted, in this case from the Halpha peak
    ha_map = cube.bin_cube(2).fit(5)
    ha_map.save_as_fits_file("gaussian_fitting/maps/computed_data/halpha/fwhm_NII.fits")


# get_turbulence_from_Halpha()

