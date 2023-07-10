from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes

from fits_analyzer import Data_cube, Map, Map_u, Map_usnr, Maps

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
    In this example, the FWHM maps of the NII cube are obtained.
    """
    nii_cube = Data_cube(fits.open("gaussian_fitting/data_cubes/night_34_wcs.fits")[0])
    # The 4 int indicates from which gaussian the FWHM will be extracted, in this case from the NII peak
    fitted_maps = nii_cube.bin_cube(2).fit_all()
    # In this case, the fit() method returns a Maps object which takes a directory to save into
    # It saves every individual map using their map.name attribute
    fitted_maps.save_as_fits_file("gaussian_fitting/maps/computed_data")


# if __name__ == "__main__":
#     get_FWHM_maps()


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
    nii_FWHM_map = Map_usnr(fits.open("gaussian_fitting/maps/computed_data_2p/NII_fwhm.fits"))
    # The pixels that have a snr inferior to 6 are masked
    nii_FWHM_map = nii_FWHM_map.filter_snr(snr_threshold=6)
    instrumental_function = Map_u(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")).bin_map(2)
    # The aligned map is the result of the subtraction of the instrumental_function map squared to the global map squared
    aligned_map = (nii_FWHM_map**2 - instrumental_function**2).align_regions()
    # The temperature maps are adjusted at the same WCS than the global maps
    temperature_map = temp_map.transfer_temperature_to_FWHM("NII").reproject_on(nii_FWHM_map)
    turbulence_map = (aligned_map - temperature_map**2)**0.5
    # The standard deviation is the desired quantity
    turbulence_map /= 2*np.sqrt(2*np.log(2))
    # turbulence_map.save_as_fits_file("gaussian_fitting/maps/computed_data_2p/turbulence.fits")
    seven_peaks_map = Map(fits.open("gaussian_fitting/maps/computed_data_2p/7_component_fit.fits"))
    temperature_map_with_7peaks = Map_usnr.from_Map_u_object(turbulence_map, seven_peaks_map)
    temperature_map_with_7peaks.save_as_fits_file("gaussian_fitting/maps/computed_data_2p/turbulence.fits")


# get_turbulence_map(Map_u(fits.HDUList([fits.open("gaussian_fitting/maps/external_maps/temp_it_nii_8300.fits")[0],
#                                        fits.open("gaussian_fitting/maps/external_maps/temp_it_nii_err_8300.fits")[0]])))


def get_courtes_temperature(settings: dict):
    """
    In this example, the pre-rendered maps are used to calculate a temperature map using Courtes's method and suppositions.
    Note that the map_1 is always reprojected onto the map_2. Which Map is subtracted to which is determined by the value of
    the subtraction key.
    """
    if settings["map_1"]["global_temperature_was_substracted"]:
        # This is the case with every Leo's maps
        map_1_FWHM = settings["map_1"]["fwhm_map"]
        temp_in_fwhm = Map.transfer_temperature_to_FWHM(Map(fits.PrimaryHDU(np.full((map_1_FWHM.data.shape), 8500), None)),
                                                        settings["map_1"]["element"])
        map_1_FWHM_with_temperature = (map_1_FWHM**2 + temp_in_fwhm**2)**0.5
        # Unnecessary data without physical significance is removed
        map_1_FWHM_with_temperature.data[map_1_FWHM_with_temperature.data > 10000] = np.NAN
    else:
        map_1_FWHM_with_temperature = settings["map_1"]["fwhm_map"]

    if settings["map_2"]["global_temperature_was_substracted"]:
        # This is the case with every Leo's maps
        map_2_FWHM = settings["map_2"]["fwhm_map"]
        temp_in_fwhm = Map.transfer_temperature_to_FWHM(Map(fits.PrimaryHDU(np.full((map_2_FWHM.data.shape), 8500), None)),
                                                        settings["map_2"]["element"])
        map_2_FWHM_with_temperature = (map_2_FWHM**2 + temp_in_fwhm**2)**0.5
        # Unnecessary data without physical significance is removed
        map_2_FWHM_with_temperature.data[map_2_FWHM_with_temperature.data > 10000] = np.NAN
    else:
        map_2_FWHM_with_temperature = settings["map_2"]["fwhm_map"]

    if settings["turbulence_consideration"]:
        turbulence_map = Map(fits.open("gaussian_fitting/maps/computed_data/turbulence.fits")) * 2*np.sqrt(2*np.log(2))
        map_1_FWHM_with_temperature = (map_1_FWHM_with_temperature**2 - 
                                       turbulence_map.reproject_on(map_1_FWHM_with_temperature)**2)**0.5
        map_2_FWHM_with_temperature = (map_2_FWHM_with_temperature**2 - 
                                       turbulence_map.reproject_on(map_2_FWHM_with_temperature)**2)**0.5

    if settings["map_1"]["fine_structure"]:
        e_factor = 2*np.sqrt(np.log(2))
        map_1_FWHM_with_temperature = (0.942 * map_1_FWHM_with_temperature / e_factor + 0.0385) * e_factor
    
    if settings["map_2"]["fine_structure"]:
        e_factor = 2*np.sqrt(np.log(2))
        map_2_FWHM_with_temperature = (0.942 * map_2_FWHM_with_temperature / e_factor + 0.0385) * e_factor

    # The FWHM maps are converted in Angstroms
    map_1_peak_AA = settings["map_1"]["peak_wavelength_AA"]
    map_2_peak_AA = settings["map_2"]["peak_wavelength_AA"]
    map_1_FWHM_with_temperature_AA = 1000 * map_1_FWHM_with_temperature * map_1_peak_AA / scipy.constants.c
    map_2_FWHM_with_temperature_AA = 1000 * map_2_FWHM_with_temperature * map_2_peak_AA / scipy.constants.c

    # The two maps are used to compute a temperature map
    if settings["subtraction"] == "1-2":
        temperature_map = 4.73 * 10**4 * (map_1_FWHM_with_temperature_AA.reproject_on(map_2_FWHM_with_temperature_AA)**2
                                          - map_2_FWHM_with_temperature_AA**2)
    elif settings["subtraction"] == "2-1":
        temperature_map = 4.73 * 10**4 * (map_2_FWHM_with_temperature_AA**2 -
                                          map_1_FWHM_with_temperature_AA.reproject_on(map_2_FWHM_with_temperature_AA)**2)

    temperature_map.plot_map((0,20000))
    # temperature_map.save_as_fits_file(settings["save_file_name"])
    double_map = Map_u.from_Map_objects(temperature_map, Map(fits.open("gaussian_fitting/maps/computed_data_2p/7_component_fit.fits")))
    double_map.save_as_fits_file(settings["save_file_name"])


# These settings allow for the computation of the temperature map using the Halpha and NII emission lines present in the NII cube
settings_Ha_NII = {
    "map_1": {"fwhm_map": (Map(fits.open("gaussian_fitting/maps/computed_data/Ha_fwhm.fits")[0])**2 - 
                           Map(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")[0]).bin_map()**2)**0.5,
              "global_temperature_was_substracted": False,
              "peak_wavelength_AA": 6562.78,
              "element": "Ha",
              "fine_structure": True},
    "map_2": {"fwhm_map": (Map(fits.open("gaussian_fitting/maps/computed_data/NII_fwhm.fits")[0])**2 -
                           Map(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")[0]).bin_map()**2)**0.5,
              "global_temperature_was_substracted": False,
              "peak_wavelength_AA": 6583.41,
              "element": "NII",
              "fine_structure": False},
    "subtraction": "1-2",
    "save_file_name": "gaussian_fitting/maps/temp_maps_courtes/new/Ha_NII.fits",
    "turbulence_consideration" : True
}

# These settings allow for the computation of the temperature map using Halpha from the NII cube and OIII from Leo's data
settings_OIII_Ha = {
    "map_1": {"fwhm_map": (Map(fits.open("gaussian_fitting/maps/new_leo/OIII.fits")[0])**2 -
                           (Map(fits.open("gaussian_fitting/leo/OIII/M2_cal.fits")[0])*8.79)**2)**0.5,
              "global_temperature_was_substracted": False,
              "peak_wavelength_AA": 5007,
              "element": "OIII",
              "fine_structure": False},
    "map_2": {"fwhm_map": (Map(fits.open("gaussian_fitting/maps/computed_data/Ha_fwhm.fits")[0])**2 - 
                           Map(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")[0]).bin_map(2)**2)**0.5,
              "global_temperature_was_substracted": False,
              "peak_wavelength_AA": 6562.78,
              "element": "Ha",
              "fine_structure": True},
    "subtraction": "2-1",
    "save_file_name": "gaussian_fitting/maps/temp_maps_courtes/turbulence_removed/OIII_Ha.fits",
    "turbulence_consideration" : True
}

# These settings allow for the computation of the temperature map using NII from the NII cube and SII from Leo's data
settings_SII_NII = {
    "map_1": {"fwhm_map": Map(fits.open("gaussian_fitting/leo/SII/SII_sigma+header.fits")[0]) * 2*np.sqrt(2*np.log(2)),
              "global_temperature_was_substracted": True,
              "peak_wavelength_AA": 6717,
              "element": "SII",
              "fine_structure": False},
    "map_2": {"fwhm_map": (Map(fits.open("gaussian_fitting/maps/computed_data_2p/NII_fwhm.fits")[0])**2 - 
                           Map(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")[0]).bin_map(2)**2)**0.5,
              "global_temperature_was_substracted": False,
              "peak_wavelength_AA": 6583.41,
              "element": "NII",
              "fine_structure": False},
    "subtraction": "2-1",
    "save_file_name": "gaussian_fitting/maps/temp_maps_courtes/new/SII_NII_2peaks.fits",
    "turbulence_consideration" : False
}


# get_courtes_temperature(settings_SII_NII)


def get_region_stats(Map, filename: str=None, write=False):
    """
    In this example, the statistics of a region are obtained and stored in a .txt file.
    """
    # Open the three studied regions
    regions = [
        None,
        pyregion.open("gaussian_fitting/regions/region_1.reg"),
        pyregion.open("gaussian_fitting/regions/region_2.reg"),
        pyregion.open("gaussian_fitting/regions/region_3.reg")
    ]
    region_names = ["Global region", "Region 1", "Region 2", "Region 3"]
    for region, region_name in zip(regions, region_names):
        stats = Map.get_region_statistics(region)
        print(stats)
        # The write bool must be set to True if the statistics need to be put in a file
        if write:
            file = open(filename, "a")
            file.write(f"{region_name}:\n")
            for key, value in stats.items():
                file.write(f"{key}: {value}\n")
            file.write("\n")
            file.close()


# get_region_stats(Map(fits.open(f"gaussian_fitting/maps/computed_data_2p/turbulence.fits")[0]), 
#                       filename="gaussian_fitting/statistics/new/turbulence_2p_stats.txt", write=True)


def get_turbulence_figure_with_regions():
    """
    In this example, the turbulence jpeg image with the regions is obtained.
    """
    turbulence_map = Map_u(fits.open("gaussian_fitting/maps/computed_data_2p/turbulence.fits"))
    # The regions need to be opened in a specific way to allow them to be juxtaposed on the turbulence map
    regions = [
        pyregion.open("gaussian_fitting/regions/region_1.reg").as_imagecoord(header=turbulence_map.header),
        pyregion.open("gaussian_fitting/regions/region_2.reg").as_imagecoord(header=turbulence_map.header),
        pyregion.open("gaussian_fitting/regions/region_3.reg").as_imagecoord(header=turbulence_map.header)
    ]

    # The following function allows for the modification of the regions' color
    def fixed_color(shape, saved_attrs):
        attr_list, attr_dict = saved_attrs
        attr_dict["color"] = "red"
        kwargs = pyregion.mpl_helper.properties_func_default(shape, (attr_list, attr_dict))
        return kwargs
    
    wcs = WCS(turbulence_map.header)
    fig = plt.figure()
    ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=wcs)
    fig.add_axes(ax)
    cbar = plt.colorbar(ax.imshow(turbulence_map.data, origin="lower"))
    patch_and_artist_list = [region.get_mpl_patches_texts(fixed_color) for region in regions]
    for region in patch_and_artist_list:
        for patch in region[0]:
            ax.add_patch(patch)
        for artist in region[1]:
            ax.add_artist(artist)
    plt.title("Turbulence de la région Sh2-158 avec un fit NII à deux composantes")
    cbar.ax.set_ylabel("turbulence (km/s)")
    plt.show()


# get_turbulence_figure_with_regions()


def get_histograms():
    """
    In this example, the histograms of the three regions and of the entire region are obtained.
    """
    turbulence_map = Map_u(fits.open("gaussian_fitting/maps/computed_data_2p/turbulence.fits"))
    regions = [
        None,
        pyregion.open("gaussian_fitting/regions/region_1.reg"),
        pyregion.open("gaussian_fitting/regions/region_2.reg"),
        pyregion.open("gaussian_fitting/regions/region_3.reg")
    ]
    histogram_names = [
        "Turbulence de la région Sh2-158",
        "Turbulence de la région diffuse de Sh2-158",
        "Turbulence de la région centrale de Sh2-158",
        "Turbulence de la région du filament de Sh2-158"
    ]
    # Note: it is possible to use the + operator between Shapelist objects to merge two regions together
    # turbulence_map.plot_region_histogram(pyregion.ShapeList(regions[1] + regions[3]), "Turbulence de la région diffuse et du filament de Sh2-158 avec un fit NII à deux composantes")
    for region, name in zip(regions, histogram_names):
        turbulence_map.plot_region_histogram(region, name)


# get_histograms()


def get_OIII_FWHM_from_Leo():
    oiii_cube = Data_cube(fits.open("gaussian_fitting/leo/OIII/reference_cube_with_header.fits")[0])
    # The OIII cube presents a single peak and may then be fitted as a calibration cube
    oiii_map = oiii_cube.fit_calibration()
    oiii_map.save_as_fits_file("gaussian_fitting/maps/new_leo/OIII.fits")


# if __name__ == "__main__":
#     get_OIII_FWHM_from_Leo()


def compare_leo_OIII():
    oiii_map = Map_u(fits.open("gaussian_fitting/maps/new_leo/OIII.fits"))
    calib_map = Map(fits.open("gaussian_fitting/leo/OIII/M2_cal.fits"))
    temp_map = Map.transfer_temperature_to_FWHM(fits.PrimaryHDU(np.full((oiii_map.data.shape), 8500), None), "OIII")

    oiii_map_filtered = (oiii_map**2 - (calib_map*8.79)**2 - temp_map**2)**0.5 / (2*np.sqrt(2*np.log(2)))
    oiii_map_filtered.save_as_fits_file("gaussian_fitting/maps/new_leo/OIII_final_chcalib.fits")


# compare_leo_OIII()


def get_turbulence_map_from_OIII(temp_map):
    oiii_FWHM_map = Map(fits.open("gaussian_fitting/maps/new_leo/OIII.fits"))
    instrumental_function = Map(fits.open("gaussian_fitting/leo/OIII/M2_cal.fits"))
    # The aligned map is the result of the subtraction of the instrumental_function map squared to the global map squared
    subtracted_map = (oiii_FWHM_map**2 - instrumental_function**2)
    # The temperature maps are adjusted at the same WCS than the global maps
    # temperature_map = temp_map.transfer_temperature_to_FWHM("NII").reproject_on(oiii_FWHM_map)
    temperature_map = Map.transfer_temperature_to_FWHM(fits.PrimaryHDU(np.full((oiii_FWHM_map.data.shape), 11000), None), "OIII")
    turbulence_map = (subtracted_map - temperature_map**2)**0.5
    # The standard deviation is the desired quantity
    turbulence_map /= 2*np.sqrt(2*np.log(2))
    turbulence_map.save_as_fits_file("gaussian_fitting/maps/new_leo/turbulence_OIII2.fits")


# get_turbulence_map_from_OIII(Map_u(fits.HDUList([fits.open("gaussian_fitting/maps/external_maps/temp_it_nii_8300.fits")[0],
#                                                  fits.open("gaussian_fitting/maps/external_maps/temp_it_nii_err_8300.fits")[0]])))


def get_temperature_from_SII_broadening():
    sii_FWHM = Map(fits.open("gaussian_fitting/leo/SII/SII_sigma+header.fits")) * 2*np.sqrt(2*np.log(2))
    turb_FWHM = Map_u(fits.open("gaussian_fitting/maps/computed_data_2p/turbulence.fits")) * 2*np.sqrt(2*np.log(2))
    global_temp = Map.transfer_temperature_to_FWHM(Map(fits.PrimaryHDU(np.full((turb_FWHM.data.shape), 8500), None)), "SII")
    temperature_FWHM_broadening = (sii_FWHM.reproject_on(turb_FWHM)**2 + global_temp**2 - turb_FWHM**2)**0.5
    # sii_FWHM.plot_map((0,40))
    # turb_FWHM.plot_map((0,40))
    # global_temp.plot_map((0,40))
    temperatures = temperature_FWHM_broadening.transfer_FWHM_to_temperature("SII")
    temperatures.plot_map((0,10**5))
    temperatures.save_as_fits_file("temp_SII.fits")
    print(temperatures.get_region_statistics(pyregion.open("gaussian_fitting/regions/region_1.reg")))

        # elements = {
        #     "NII":  {"emission_peak": 6583.41, "mass_u": 14.0067},
        #     "Ha":   {"emission_peak": 6562.78, "mass_u": 1.00784},
        #     "SII":  {"emission_peak": 6716,    "mass_u": 32.065},
        #     "OIII": {"emission_peak": 5007,    "mass_u": 15.9994}
        # }
        # angstroms_center = elements[element]["emission_peak"]     # Emission wavelength of the element
        # m = elements[element]["mass_u"] * scipy.constants.u         # Mass of the element
        # c = scipy.constants.c                                     # Light speed
        # k = scipy.constants.k                                     # Boltzmann constant
        # angstroms_FWHM = 2 * float(np.sqrt(2 * np.log(2))) * angstroms_center * (self * k / (c**2 * m))**0.5
        # speed_FWHM = c * angstroms_FWHM / angstroms_center / 1000


# get_temperature_from_SII_broadening()


