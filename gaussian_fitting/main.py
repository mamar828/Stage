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


def get_maps():
    """
    In this example, all important maps that will be used later are computed once.
    """
    nii_cube = Data_cube(fits.open("gaussian_fitting/data_cubes/night_34_wcs.fits")[0])
    # The cube is binned, then fitted, and the FWHM, mean and amplitude of every gaussian is stored
    # The extract argument specifies the order in which the Maps will be returned
    # Note that the extract argument can have fewer elements if not all Maps are desired
    fwhm_maps, mean_maps, amplitude_maps = nii_cube.bin_cube(2).fit_NII_cube(extract=["FWHM", "mean", "amplitude"])
    # All fwhm_maps are saved
    fwhm_maps.save_as_fits_file("gaussian_fitting/maps/computed_data")
    # Only the NII and Ha maps for mean and amplitude are saved
    mean_maps["NII_mean"].save_as_fits_file("gaussian_fitting/maps/computed_data_test/NII_mean.fits")
    mean_maps["Ha_mean"].save_as_fits_file("gaussian_fitting/maps/computed_data_test/Ha_mean.fits")
    amplitude_maps["NII_amplitude"].save_as_fits_file("gaussian_fitting/maps/computed_data_test/NII_amplitude.fits")
    amplitude_maps["Ha_amplitude"].save_as_fits_file("gaussian_fitting/maps/computed_data_test/Ha_amplitude.fits")


# Note that some functions are called in a if __name__ == "__main__" block because the fitting algorithm uses the multiprocessing
# library which creates multiple instances of the same code to allow parallel computation. Without this condition, the program would
# multiply itself recursively.
# if __name__ == "__main__":
#     get_maps()


def example_fitting():
    """
    This example is never used but is intended to clarify the fitting method and the returned objects.
    """
    nii_cube = Data_cube(fits.open("gaussian_fitting/data_cubes/night_34_wcs.fits")[0])
    # The extract argument controls what will be returned and in what order by the fit_all() method
    mean_maps, amplitude_maps = nii_cube.fit_all(extract=["mean", "amplitude"])
    # The fit_all() method returns Maps objects which are a list of Map objects and its subclasses
    # Each_map can be accessed as done below
    mean_maps["OH1_mean"].plot_map()            # The first OH peak's mean value is plotted
    amplitude_maps["Ha_amplitude"].plot_map()   # The Ha peak's amplitude value is plotted
    mean_maps["7_components_fit"].plot_map()    # The map that tells if a double NII peak was detected is plotted
    # It is also possible to save an entire Maps object into a folder
    mean_maps.save_as_fits_file("path/to/folder")

    # Two NII gaussians can also be authorized for fitting
    fwhm_maps = nii_cube.fit_all(extract=["FWHM"], seven_components_fit_authorized=True)
    fwhm_maps.save_as_fits_file("path/to/folder")


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


def get_NII_flux_maps():
    """
    In this example, the map of the multiplication of the intensity by the channel spacing is obtained. This corresponds to M0 which
    is equation 2.
    The NII intensity-weighted centroid velocity map is also obtained. This correspond to M1 which is equation 3.
    """
    nii_cube = Data_cube(fits.open("gaussian_fitting/data_cubes/night_34_wcs.fits"))
    speed_per_channel = nii_cube.header["CDELT3"]
    speed_channel_1 = nii_cube.header["CRVAL3"]
    spectral_length = nii_cube.header["FP_I_A"]
    wavelength_channel_1 = nii_cube.header["FP_B_L"]
    number_of_channels = nii_cube.header["NAXIS3"]
    
    # The NII gaussian first needs to be reconstructed
    amplitude = Map_u(fits.open("gaussian_fitting/maps/computed_data/NII_amplitude.fits"))
    mean = Map_u(fits.open("gaussian_fitting/maps/computed_data/NII_mean.fits"))
    fwhm = Map_u(fits.open("gaussian_fitting/maps/computed_data/NII_fwhm.fits"))
    angstroms_center = mean * spectral_length / number_of_channels + wavelength_channel_1
    fwhm_channels = (fwhm*1000 / scipy.constants.c * angstroms_center) / spectral_length * number_of_channels
    sigma = fwhm_channels / (2*np.sqrt(2*np.log(2)))

    # To calculate the NII gaussian's intensity, the gaussian function equation may be used but all arrays must have the same shape
    a = amplitude.add_new_axis(48)
    x0 = mean.add_new_axis(48)
    s = sigma.add_new_axis(48)

    # The array which will sample the x values is created
    channel_numbers = np.linspace(1, 48, 48)
    channel_numbers = np.tile(channel_numbers, 512*512).reshape(512,512,48)
    intensity = a * np.e**(-(channel_numbers - x0)**2 / (2 * s**2))

    flux_data = np.nansum(intensity.data, 2)
    flux = Map(fits.PrimaryHDU(flux_data, amplitude.header)) * speed_per_channel
    flux.save_as_fits_file("gaussian_fitting/maps/computed_data/flux_map.fits")

    # The intensity weighted velocity is computed here
    speed_at_every_channel = speed_channel_1 + (channel_numbers - 1) * speed_per_channel
    product = speed_at_every_channel * intensity * speed_per_channel
    numerator_data = np.nansum(product.data, 2)
    numerator = Map(fits.PrimaryHDU(numerator_data, amplitude.header))
    intensity_weighted_centroid_velocity = numerator / flux
    intensity_weighted_centroid_velocity.save_as_fits_file("gaussian_fitting/maps/computed_data/intensity_weighted_velocity.fits")


# get_NII_flux_maps()


def get_NII_Doppler_shift():
    """
    In this example, the NII Doppler shift is computed.
    """
    # The NII cube is opened so spectral data provided in its header can be used
    nii_cube = Data_cube(fits.open("gaussian_fitting/data_cubes/night_34_wcs.fits")[0])
    mean_map = Map_u(fits.open("gaussian_fitting/maps/computed_data/NII_mean.fits"))
    spectral_length = nii_cube.header["FP_I_A"]
    wavelength_channel_1 = nii_cube.header["FP_B_L"]
    number_of_channels = nii_cube.header["NAXIS3"]
    mean_map_AA = mean_map * spectral_length / number_of_channels + wavelength_channel_1
    doppler_shift = (mean_map_AA - 6583.41) / 6583.41 * scipy.constants.c / 1000
    doppler_shift.save_as_fits_file("gaussian_fitting/maps/computed_data/NII_doppler_shift.fits")


# get_NII_Doppler_shift()


def get_region_widening_maps(base_map: Map_usnr):
    """
    In this example, the four headers are extracted and attributed to the various region widenings. The data is stored in
    gaussian_fitting/maps/reproject. These maps are useful only for their headers which locally correct for the data cube's
    distortion.
    """
    global_header = Data_cube(fits.open("gaussian_fitting/data_cubes/night_34_wcs.fits")[0]).header
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
    nii_FWHM_map = Map_usnr(fits.open("gaussian_fitting/maps/computed_data/NII_fwhm.fits"))
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
    turbulence_map.save_as_fits_file("gaussian_fitting/maps/computed_data/turbulence.fits")


# get_turbulence_map(Map_u(fits.HDUList([fits.open("gaussian_fitting/maps/external_maps/temp_it_nii_8300.fits")[0],
#                                        fits.open("gaussian_fitting/maps/external_maps/temp_it_nii_err_8300.fits")[0]])))


def get_courtes_temperature(settings: dict):
    """
    In this example, the pre-rendered maps are used to calculate a temperature map using Courtes's method and suppositions.
    This example is made so it can be used with any map and only a settings dict needs to be changed. In the settings dict, the
    different keys have the following use:
    "map_1": informations of the first map which will be projected onto the second_map in the form of a dict
        "fwhm_map": Map object. First map on which the following settings will apply.
        "global_temperature_was_substracted": bool. If True, the broadening associated to a temperature of 8500K will be added to
            the current broadening.
        "peak_wavelength_AA": int. Wavelength of the element's emission peak in Angstroms.
        "element": str. Name of the element whose broadening is present in the map. This is used to add a global temperature if the
            "global_temperature_was_substracted" bool is set to True. Supported names are "NII", "Ha", "OIII" and "SII".
        "fine_structure": bool. If True, the broadening associated to the fine structure in the hydrogen atom will be substracted
            from the map.
    "map_2": informations on the second map. It has the same format as map_1.
    "subtraction": str. Specifies which map to subtract to which map. The map which has the heaviest element should always be the 
        one subtracting the other as heavier elements have smaller ray broadening. The str format is "1-2" or "2-1".
    "turbulence_consideration: bool. If True, the broadening associated to the turbulence map will be subtracted from each map.
        This is primordial when the fine_structure of either map is set to True as the formula considers that the broadening does
        not have any turbulent nature.
    "save_file_name": str. Name of the file to which the temperature map will be saved.
    """
    if settings["map_1"]["global_temperature_was_substracted"]:
        # This is the case with every map of Leo
        map_1_FWHM = settings["map_1"]["fwhm_map"]
        # A map of the shape of the present map representing the broadening due to thermal effects at every pixel is created
        temp_in_fwhm = Map.transfer_temperature_to_FWHM(Map(fits.PrimaryHDU(np.full((map_1_FWHM.data.shape), 8500), None)),
                                                        settings["map_1"]["element"])
        map_1_FWHM_with_temperature = (map_1_FWHM**2 + temp_in_fwhm**2)**0.5
        # Unnecessary data without physical significance is removed, in this case pixels with a FWHM superior to 10 000
        map_1_FWHM_with_temperature.data[map_1_FWHM_with_temperature.data > 10000] = np.NAN
    else:
        map_1_FWHM_with_temperature = settings["map_1"]["fwhm_map"]

    if settings["map_2"]["global_temperature_was_substracted"]:
        # This is the case with every map of Leo
        map_2_FWHM = settings["map_2"]["fwhm_map"]
        # A map of the shape of the present map representing the broadening due to thermal effects at every pixel is created
        temp_in_fwhm = Map.transfer_temperature_to_FWHM(Map(fits.PrimaryHDU(np.full((map_2_FWHM.data.shape), 8500), None)),
                                                        settings["map_2"]["element"])
        map_2_FWHM_with_temperature = (map_2_FWHM**2 + temp_in_fwhm**2)**0.5
        # Unnecessary data without physical significance is removed, in this case pixels with a FWHM superior to 10 000
        map_2_FWHM_with_temperature.data[map_2_FWHM_with_temperature.data > 10000] = np.NAN
    else:
        map_2_FWHM_with_temperature = settings["map_2"]["fwhm_map"]

    if settings["turbulence_consideration"]:
        # The turbulence broadening is removed
        turbulence_map = Map(fits.open("gaussian_fitting/maps/computed_data/turbulence.fits")) * 2*np.sqrt(2*np.log(2))
        map_1_FWHM_with_temperature = (map_1_FWHM_with_temperature**2 - 
                                       turbulence_map.reproject_on(map_1_FWHM_with_temperature)**2)**0.5
        map_2_FWHM_with_temperature = (map_2_FWHM_with_temperature**2 - 
                                       turbulence_map.reproject_on(map_2_FWHM_with_temperature)**2)**0.5

    if settings["map_1"]["fine_structure"]:
        # The fine structure broadening is removed
        e_factor = 2*np.sqrt(np.log(2))
        map_1_FWHM_with_temperature = (0.942 * map_1_FWHM_with_temperature / e_factor + 0.0385) * e_factor
    
    if settings["map_2"]["fine_structure"]:
        # The fine structure broadening is removed
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
    temperature_map.save_as_fits_file(settings["save_file_name"])

"""
# These se ttings allow for the computation of the temperature map using the Halpha and NII emission lines present in the NII cube
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
    "turbulence_consideration" : True,
    "save_file_name": "gaussian_fitting/maps/temp_maps_courtes/new/Ha_NII.fits"
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
    "turbulence_consideration" : True,
    "save_file_name": "gaussian_fitting/maps/temp_maps_courtes/turbulence_removed/OIII_Ha.fits"
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
    "turbulence_consideration" : False,
    "save_file_name": "gaussian_fitting/maps/temp_maps_courtes/new/SII_NII_2peaks.fits"
}
"""

# get_courtes_temperature(settings_SII_NII)


def get_region_stats(Map, filename: str=None, write=False):
    """
    In this example, the statistics of a map are printed and stored in a .txt file.
    """
    # Open the three studied regions
    # The fact that the first element is None allows the stats to be calculated on the entire region
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
#                  filename="gaussian_fitting/statistics/new/turbulence_2p_stats.txt", write=True)


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
    
    fig = plt.figure()
    # The axes are set to have celestial coordinates
    wcs = WCS(turbulence_map.header)
    ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=wcs)
    fig.add_axes(ax)
    # The turbulence map is plotted along with the colorbar, inverting the axes so y=0 is at the bottom
    plot_with_cbar = plt.colorbar(ax.imshow(turbulence_map.data, origin="lower"))
    # The data of every region is exported in a format usable by matplotlib
    patch_and_artist_list = [region.get_mpl_patches_texts(fixed_color) for region in regions]
    # The regions are placed on the map
    for region in patch_and_artist_list:
        for patch in region[0]:
            ax.add_patch(patch)
        for artist in region[1]:
            ax.add_artist(artist)
    plt.title("Turbulence de la région Sh2-158 avec un fit NII à deux composantes")
    plot_with_cbar.ax.set_ylabel("turbulence (km/s)")
    plt.show()


# get_turbulence_figure_with_regions()


def get_histograms():
    """
    In this example, the different histograms of the turbulence map are obtained.
    """
    turbulence_map = Map_u(fits.open("gaussian_fitting/maps/computed_data_2p/turbulence.fits"))
    # The first element in the regions list is None because this allows the statistics to be calculated over the entire region
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
    # Note: it is possible to use the + operator between Shapelist objects to merge two regions together as done below
    # turbulence_map.plot_region_histogram(pyregion.ShapeList(regions[1] + regions[3]), "Turbulence de la région diffuse et du...")
    for region, name in zip(regions, histogram_names):
        turbulence_map.plot_region_histogram(region, name)


# get_histograms()


def get_OIII_FWHM_from_Leo():
    """
    In this example, the OIII FWHM is obtained through Leo's raw cube.
    """
    oiii_cube = Data_cube(fits.open("gaussian_fitting/leo/OIII/reference_cube_with_header.fits")[0])
    # The OIII cube presents a single peak and may then be fitted as a calibration cube
    oiii_map = oiii_cube.fit_calibration()
    oiii_map.save_as_fits_file("gaussian_fitting/maps/new_leo/OIII.fits")


# if __name__ == "__main__":
#     get_OIII_FWHM_from_Leo()


def get_temperature_from_SII_broadening():
    """
    In this example, a temperature map is obtained from the broadening of the singly ionized sulfur atom.
    """
    sii_FWHM = Map(fits.open("gaussian_fitting/leo/SII/SII_sigma+header.fits")) * 2*np.sqrt(2*np.log(2))
    turb_FWHM = Map_u(fits.open("gaussian_fitting/maps/computed_data_2p/turbulence.fits")) * 2*np.sqrt(2*np.log(2))
    # A map of the shape of the turbulence map representing the broadening of SII caused by temperature is generated
    global_temp = Map.transfer_temperature_to_FWHM(Map(fits.PrimaryHDU(np.full((turb_FWHM.data.shape), 8500), None)), "SII")
    temperature_FWHM_broadening = (sii_FWHM.reproject_on(turb_FWHM)**2 + global_temp**2 - turb_FWHM**2)**0.5
    temperatures = temperature_FWHM_broadening.transfer_FWHM_to_temperature("SII")
    temperatures.plot_map((0,20000))
    # The following map has never been saved because of the results it gives
    temperatures.save_as_fits_file("temp_SII.fits")


# get_temperature_from_SII_broadening()


def get_ACF_plot(calc=False):
    """
    In this example, the ACF is plotted for different steps. If calc is True, the function is calculated but not plotted.
    """
    step_range = np.round(np.arange(0.1,1.6,0.1), 1)
    if calc:
        turbulence_map = Map_u(fits.open("gaussian_fitting/maps/computed_data/turbulence.fits"))
        step = None
        print("Current bin:", step)
        np.save(f"gaussian_fitting/arrays_u/turbulence_map/ACF/bin={step}.npy", turbulence_map.get_autocorrelation_function_array(step))
        for step in step_range:
            print("Current bin:", step)
            np.save(f"gaussian_fitting/arrays_u/turbulence_map/ACF/bin={step}.npy", turbulence_map.get_autocorrelation_function_array(step))
    else:
        step = None
        data_array = np.load(f"gaussian_fitting/arrays/turbulence_map/ACF/bin={step}.npy", allow_pickle=True)
        plt.plot(data_array[:,0], data_array[:,1], "mo", markersize=1)
        plt.title(step)
        plt.show()
        for step in step_range:
            data_array = np.load(f"gaussian_fitting/arrays/turbulence_map/ACF/bin={step}.npy", allow_pickle=True)
            plt.plot(data_array[:,0], data_array[:,1], "mo", markersize=1)
            plt.title(step)
            plt.show()


# if __name__ == "__main__":
    # turbulence_map = Map(fits.open("gaussian_fitting/maps/computed_data/turbulence.fits")[0])
    # step = 0.7
    # print("Current bin:", step)
    # np.save(f"gaussian_fitting/arrays/turbulence_map/ACF/bin={step}.npy", turbulence_map.get_autocorrelation_function_array(step))
    # d = np.load("gaussian_fitting/bin=0.1.npy")
    # plt.plot(d[:,0], d[:,1], "mo", markersize=1)

    # plt.show()

    # get_ACF_plot(calc=True)
    # get_ACF_plot()


def get_structure_function_plot(calc=False):
    """
    In this example, the structure function is plotted for different steps. If calc is True, the function is calculated
    but not plotted.
    """
    step_range = np.round(np.arange(0.1,1.6,0.1), 1)
    if calc:
        turbulence_map = Map_u(fits.open("gaussian_fitting/maps/computed_data/turbulence.fits"))
        step = None
        print("Current bin:", step)
        np.save(f"gaussian_fitting/arrays_u/turbulence_map/structure_function/bin={step}.npy", 
                turbulence_map.get_structure_function_array(step))
        for step in step_range:
            print("Current bin:", step)
            np.save(f"gaussian_fitting/arrays_u/turbulence_map/structure_function/bin={step}.npy", 
                    turbulence_map.get_structure_function_array(step))
    else:
        step = None
        data_array = np.load(f"gaussian_fitting/arrays/turbulence_map/structure_function/bin={step}.npy", allow_pickle=True)
        plt.plot(data_array[:,0], data_array[:,1], "mo", markersize=1)
        plt.title(step)
        plt.show()
        for step in step_range:
            data_array = np.load(f"gaussian_fitting/arrays/turbulence_map/structure_function/bin={step}.npy", allow_pickle=True)
            plt.plot(data_array[:,0], data_array[:,1], "mo", markersize=1)
            plt.title(step)
            plt.show()


# data_array = np.load(f"gaussian_fitting/arrays/turbulence_map/structure_function/ints.npy", allow_pickle=True)
# plt.plot(data_array[:,0], data_array[:,1], "mo", markersize=1)
# plt.show()

# if __name__ == "__main__":
#     get_structure_function_plot(calc=True)


def test_structure():
    turbulence_map = Map(fits.open("gaussian_fitting/maps/computed_data/turbulence.fits")[0])
    turbulence_map.test_structure_func()
    # np.save(f"gaussian_fitting/test_1_lag.npy", turbulence_map.test_structure_func())
    # data_array = np.load(f"gaussian_fitting/data_arrays/data_array_b{step}.npy", allow_pickle=True)
    # plt.plot(data_array[:,0], data_array[:,1], "mo", markersize=1)
    # plt.show()


# if __name__ == "__main__":
#     test_structure()


def get_fit_function(array, s_factor):
    """
    In this example, the data of a certain statistical method is fitted using an spline, and the data, spline and spline derivative
    are plotted.
    """
    spline = scipy.interpolate.splrep(array[:,0], array[:,1], s=s_factor)
    plt.plot(array[:,0], array[:,1], "ro", markersize=1, label="Data")
    x_sample = np.linspace(1, array[-1,0], 10000)
    plt.plot(x_sample, scipy.interpolate.BSpline(*spline)(x_sample), "g", label=f"Fitted spline with s={s_factor}")
    plt.plot(x_sample, scipy.interpolate.BSpline(*spline)(x_sample, 1), "y", label=f"Fitted spline's derivative")
    plt.legend()
    plt.show()


# get_fit_function(np.load("gaussian_fitting/arrays/turbulence_map/structure_function/bin=1.2.npy"), 15)


# if __name__ == "__main__":
#     nii_centroid = Map_u(fits.open("gaussian_fitting/maps/computed_data/NII_mean.fits"))
#     nii_fwhm = Map_usnr(fits.open("gaussian_fitting/maps/computed_data/NII_fwhm.fits"))
#     nii_centroid_snr = Map_usnr.from_Map_u_object(nii_centroid, nii_fwhm[2])
#     filtered_nii = nii_centroid_snr.filter_snr(6)[130:380,70:410]
#     filtered_nii.plot_map()
#     np.save("gaussian_fitting/bin=None.npy", filtered_nii.get_autocorrelation_function_array())
#     # np.save("gaussian_fitting/arrays/nii_centroid_map/ACF/bin=None.npy", filtered_nii.get_autocorrelation_function_array())
#     data_array = np.load(f"gaussian_fitting/arrays/nii_centroid_map/ACF/bin=None.npy", allow_pickle=True)
#     plt.plot(data_array[:,0], data_array[:,1], "mo", markersize=1)
#     plt.show()


# for i in np.round(np.arange(1.2,1.6,0.1), 1):
#     old = np.load(f"gaussian_fitting/bin={i}.npy")
#     new = np.load(f"gaussian_fitting/arrays_u/turbulence_map/structure_function/bin={i}.npy")
#     ax1 = plt.subplot(1,2,1)
#     ax2 = plt.subplot(1,2,2)
#     ax1.plot(old[:,0], old[:,1], "o", markersize=1)
#     ax2.plot(new[:,0], new[:,1], "o", markersize=1)
#     plt.title(i)
#     plt.show()
#     print(i, np.max(old[:,1] - new[:,1]))

