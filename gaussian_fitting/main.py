from astropy.io import fits
from astropy.wcs import WCS

from fits_analyzer import Data_cube, Map, Map_u, Map_usnr

import matplotlib.pyplot as plt
import numpy as np
import scipy

import pyregion

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
    In this example, the FWHM_NII map is obtained.
    """
    nii_cube = Data_cube(fits.open("gaussian_fitting/data_cubes/night_34_wcs.fits")[0])
    # The 4 int indicates from which gaussian the FWHM will be extracted, in this case from the NII peak
    nii_map = nii_cube.bin_cube(2).fit(4, True)
    nii_map.save_as_fits_file("gaussian_fitting/maps/computed_data/fwhm_NII.fits")


if __name__ == "__main__":
    get_FWHM_maps()


def get_region_widening_maps(base_map: Map_usnr):
    """
    In this example, the four headers are extracted and attributed to the various region widenings. The data is 
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

        data_map = Map_usnr(fits.HDUList([fits.PrimaryHDU(base_map.data, header_copy),
                                          fits.ImageHDU(base_map.uncertainties, header_copy),
                                          fits.ImageHDU(base_map.snr, header_copy)]))
        data_map.save_as_fits_file(f"gaussian_fitting/maps/reproject/{filename}")


# get_region_widening_maps(Map_usnr(fits.open("gaussian_fitting/maps/computed_data/fwhm_NII.fits")))


def get_turbulence_map(temp_map):
    """
    In this example, the turbulence map is obtained with the previously computed maps: all region widenings as well as
    smoothed_instr_f. Note that the region widenings maps are not opened directly but are used in the Map.align_regions() method.
    """
    global_FWHM_map = Map_usnr(fits.open("gaussian_fitting/maps/reproject/global_widening.fits"))
    # The pixels that have a snr inferior to 6 are masked
    global_FWHM_map = global_FWHM_map.filter_snr(snr_threshold=6)
    instrumental_function = Map_u(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")).bin_map(2)
    # The aligned map is the result of the subtraction of the instrumental_function map squared to the global map squared
    aligned_map = (global_FWHM_map**2 - instrumental_function**2).align_regions()
    (aligned_map**0.5).save_as_fits_file("gaussian_fitting/maps/computed_data/fwhm_NII-calib.fits")
    # The temperature maps are adjusted at the same WCS than the global maps
    temperature_map = temp_map.transfer_temperature_to_FWHM().reproject_on(global_FWHM_map)
    turbulence_map = (aligned_map - temperature_map**2)**0.5
    # The standard deviation is the desired quantity
    turbulence_map /= 2 * np.sqrt(2 * np.log(2))
    turbulence_map.plot_map()
    turbulence_map.save_as_fits_file("gaussian_fitting/maps/computed_data/turbulence.fits")


# get_turbulence_map(Map_u(fits.HDUList([fits.open("gaussian_fitting/maps/external_maps/temp_it_nii_8300.fits")[0],
#                                        fits.open("gaussian_fitting/maps/external_maps/temp_it_nii_err_8300.fits")[0]])))


def get_courtes_temperature_from_NII_and_SII():
    """
    In this example, we obtain a temperature map using Courtes's method with the NII and SII emission lines.
    """
    # Here the SII emission line sigma is obtained (includes the temperature's contribution)
    # A global temperature of 8500K was used
    sii_FWHM = Map(fits.open("gaussian_fitting/leo/SII/SII_sigma+header.fits")[0]) * 2*np.sqrt(2*np.log(2))
    temp_in_fwhm = Map.transfer_temperature_to_FWHM(fits.PrimaryHDU(np.full((sii_FWHM.data.shape), 8500), None))
    sii_FWHM_with_temperature = (sii_FWHM**2 + temp_in_fwhm**2)**0.5
    sii_FWHM_with_temperature.data[sii_FWHM_with_temperature.data > 10000] = np.NAN
    sii_sigma_with_temperature = sii_FWHM_with_temperature / (2 * np.sqrt(2 * np.log(2)))

    # The NII sigma map is acquired
    nii_FWHM_with_temperature = Map(fits.open("gaussian_fitting/maps/computed_data/fwhm_NII-calib.fits")[0])
    nii_sigma_with_temperature = nii_FWHM_with_temperature / (2 * np.sqrt(2 * np.log(2)))

    # The FWHM maps are converted in Angstroms
    sii_peak_AA = 6716
    nii_peak_AA = 6583.41
    
    sii_sigma_with_temperature_AA = 1000 * sii_sigma_with_temperature * sii_peak_AA / scipy.constants.c
    nii_sigma_with_temperature_AA = 1000 * nii_sigma_with_temperature * nii_peak_AA / scipy.constants.c

    # The two maps are used to compute a temperature map
    temperature_map = 4.73 * 10**4 * (nii_sigma_with_temperature_AA**2 - 
                       sii_sigma_with_temperature_AA.reproject_on(nii_sigma_with_temperature_AA)**2)
    temperature_map.save_as_fits_file("gaussian_fitting/maps/temp_maps_courtes/NII_SII.fits")


# get_courtes_temperature_from_NII_and_SII()


def get_courtes_temperature_from_NII_and_Halpha():
    """
    In this example, we obtain a temperature map using Courtes's method with the NII and Halpha emission lines.
    """
    # Here the Halpha emission line sigma is obtained (includes the temperature's contribution)
    halpha_FWHM_with_temperature = (Map(fits.open("gaussian_fitting/maps/computed_data/fwhm_Halpha.fits")[0])**2 - 
                                    Map(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")[0]).bin_map(2)**2)**0.5
    halpha_sigma_with_temperature = halpha_FWHM_with_temperature / (2 * np.sqrt(2 * np.log(2)))

    # The NII sigma map is acquired
    nii_FWHM_with_temperature = Map(fits.open("gaussian_fitting/maps/computed_data/fwhm_NII-calib.fits")[0])
    nii_sigma_with_temperature = nii_FWHM_with_temperature / (2 * np.sqrt(2 * np.log(2)))

    # The FWHM maps are converted in Angstroms
    halpha_peak_AA = 6562.78
    nii_peak_AA = 6583.41
    
    halpha_sigma_with_temperature_AA = 1000 * halpha_sigma_with_temperature * halpha_peak_AA / scipy.constants.c
    nii_sigma_with_temperature_AA = 1000 * nii_sigma_with_temperature * nii_peak_AA / scipy.constants.c

    # The two maps are used to compute a temperature map
    temperature_map = 4.73 * 10**4 * (halpha_sigma_with_temperature_AA.reproject_on(nii_sigma_with_temperature_AA)**2 - 
                       nii_sigma_with_temperature_AA**2)
    temperature_map.save_as_fits_file("gaussian_fitting/maps/temp_maps_courtes/NII_Halpha.fits")


# get_courtes_temperature_from_NII_and_Halpha()


def get_courtes_temperature_from_NII_and_Halpha2():
    """
    In this example, we obtain a temperature map using Courtes's method with the NII and Halpha emission lines.
    """
    # Here the Halpha emission line sigma is obtained (includes the temperature's contribution)
    halpha_FWHM_with_temperature = Map(fits.open("gaussian_fitting/maps/computed_data/fwhm_Halpha.fits")[0])
    halpha_sigma_with_temperature = halpha_FWHM_with_temperature / (2 * np.sqrt(2 * np.log(2)))

    # The NII sigma map is acquired
    nii_FWHM_with_temperature = Map(fits.open("gaussian_fitting/maps/computed_data/fwhm_NII.fits")[0])
    nii_sigma_with_temperature = nii_FWHM_with_temperature / (2 * np.sqrt(2 * np.log(2)))

    # The FWHM maps are converted in Angstroms
    halpha_peak_AA = 6562.78
    nii_peak_AA = 6583.41
    
    halpha_sigma_with_temperature_AA = 1000 * halpha_sigma_with_temperature * halpha_peak_AA / scipy.constants.c
    nii_sigma_with_temperature_AA = 1000 * nii_sigma_with_temperature * nii_peak_AA / scipy.constants.c

    # The two maps are used to compute a temperature map
    temperature_map = 4.73 * 10**4 * (halpha_sigma_with_temperature_AA.reproject_on(nii_sigma_with_temperature_AA)**2 - 
                       nii_sigma_with_temperature_AA**2)
    temperature_map.plot_map((0,10000))
    temperature_map.save_as_fits_file("gaussian_fitting/maps/temp_maps_courtes/NII_Halpha2.fits")


# get_courtes_temperature_from_NII_and_Halpha2()


def get_courtes_temperature_from_Halpha_and_OIII():
    """
    In this example, we obtain a temperature map using Courtes's method with the NII and OIII emission lines.
    """
    # Here the OIII emission line sigma is obtained (includes the temperature's contribution)
    # A global temperature of 8500K was used
    oiii_FWHM = Map(fits.open("gaussian_fitting/leo/OIII/OIII_sigma+header.fits")[0]) * 2*np.sqrt(2*np.log(2))
    temp_in_fwhm = Map.transfer_temperature_to_FWHM(fits.PrimaryHDU(np.full((oiii_FWHM.data.shape), 8500), None))
    oiii_FWHM_with_temperature = (oiii_FWHM**2 + temp_in_fwhm**2)**0.5
    oiii_FWHM_with_temperature.data[oiii_FWHM_with_temperature.data > 10000] = np.NAN
    oiii_sigma_with_temperature = oiii_FWHM_with_temperature / (2 * np.sqrt(2 * np.log(2)))

    # The Halpha sigma map is acquired
    halpha_FWHM_with_temperature = (Map(fits.open("gaussian_fitting/maps/computed_data/fwhm_Halpha.fits")[0])**2 - 
                                    Map(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")[0]).bin_map(2)**2)**0.5
    halpha_sigma_with_temperature = halpha_FWHM_with_temperature / (2 * np.sqrt(2 * np.log(2)))

    # The FWHM maps are converted in Angstroms
    oiii_peak_AA = 5007
    halpha_peak_AA = 6562.78
    
    oiii_sigma_with_temperature_AA = 1000 * oiii_sigma_with_temperature * oiii_peak_AA / scipy.constants.c
    halpha_sigma_with_temperature_AA = 1000 * halpha_sigma_with_temperature * halpha_peak_AA / scipy.constants.c

    # The two maps are used to compute a temperature map
    temperature_map = 4.73 * 10**4 * (halpha_sigma_with_temperature_AA**2 - 
                       oiii_sigma_with_temperature_AA.reproject_on(halpha_sigma_with_temperature_AA)**2)
    temperature_map.save_as_fits_file("gaussian_fitting/maps/temp_maps_courtes/Halpha_OIII.fits")


# get_courtes_temperature_from_Halpha_and_OIII()


def get_region_statistics(map, write=False):
    """
    In this example, the statistics of a region are obtained and stored in the turbulence_stats.txt file.
    """
    # Open the three possible regions
    regions = [
        pyregion.open("gaussian_fitting/regions/region_1.reg"),
        pyregion.open("gaussian_fitting/regions/region_2.reg"),
        pyregion.open("gaussian_fitting/regions/region_3.reg")
    ]
    for i, region in enumerate(regions):
        # A histogram may be shown if the plot_histogram bool is set to True
        stats = map.get_region_statistics(region, plot_histogram=False)
        print(stats)
        plt.show()
        # The write bool must be set to True if the statistics need to be put in the file
        if write:
            file = open("gaussian_fitting/maps/temp_maps_courtes/NII_Halpha2.txt", "a")
            file.write(f"Region {i+1}:\n")
            for key, value in stats.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")
            file.close()


# get_region_statistics(Map(fits.open(f"gaussian_fitting/maps/temp_maps_courtes/NII_Halpha2.fits")[0]), write=True)

