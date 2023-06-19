import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
import scipy

from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import pyregion

from cube_spectrum import Spectrum

import multiprocessing
import time

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class Fits_file():
    """
    Encapsulate the methods specific to .fits files that can be used both in data cube and map analysis.
    """

    def bin_header(self, nb_pix_bin=2):
        """
        Bin the header to make the WCS match with a binned map.
        Note that this method only works if the binned map has not been cropped in the binning process. Otherwise, the WCS will
        not match.

        Arguments
        ---------
        nb_pix_bin: int. Specifies the number of pixels to be binned together along a single axis.

        Returns
        -------
        astropy.io.fits.header.Header: binned header.
        """
        header_copy = self.header.copy()
        # The try statement makes it so calibration maps/cubes can also be binned
        try:
            header_copy["CDELT1"] *= nb_pix_bin
            header_copy["CDELT2"] *= nb_pix_bin
            header_copy["CRPIX1"] /= nb_pix_bin
            header_copy["CRPIX2"] /= nb_pix_bin
        except:
            pass
        return header_copy
    
    def save_as_fits_file(self, filename=str):
        """
        Write an array as a fits file of the specified name with or without a header. If the object has a header, it will be saved.

        Arguments
        ---------
        filename: str. Indicates the path and name of the created file. If the file already exists, it is overwritten.
        header: astropy.io.fits.header.Header, optional. If specified, the fits file will have the given header. This is mainly
        useful for saving maps with usable WCS.
        """
        # Check if the file already exists
        try:
            fits.open(filename)[0]
            # The file already exists
            while True:
                answer = input(f"The file '{filename}' already exists, do you wish to overwrite it ? [y/n]")
                if answer == "y":
                    fits.writeto(filename, self.data, self.header, overwrite=True)
                    print("File overwritten.")
                    break
                elif answer == "n":
                    break        
                
        except:
            # The file does not yet exist
            fits.writeto(filename, self.data, self.header, overwrite=True)

    def reset_update_file(self):
        """
        Reset the update output file. If the file does not yet exist, it is created. This method should always be called before
        a loop.
        """
        file = open("output.txt", "w")
        file.write("0")
        file.close()

    def give_update(self, info=str):
        """
        Give the user an update of the status of the running code in the text file output.txt.

        Arguments
        ---------
        info: str. Beggining string to give information about the current running program.
        """
        file = open("output.txt", "r")
        number = int(file.readlines()[-1])
        file = open("output.txt", "w")
        file.write(f"{info}\n{str(number + 1)}")
        file.close()



class Data_cube(Fits_file):
    """
    Encapsulate all the useful methods for the analysis of a data cube.
    """

    def __init__(self, fits_object):
        """
        Initialize an analyzer object. The datacube's file name must be given.

        Arguments
        ---------
        fits_object: astropy.io.fits.hdu.image.PrimaryHDU. Contains the data values and header of the data cube.
        """
        self.object = fits_object
        self.data = fits_object.data
        self.header = fits_object.header

    def fit_calibration(self):
        """
        Fit the whole data cube as if it was a calibration cube and extract the FWHM and its uncertainty at every point.
        WARNING: Due to the use of the multiprocessing library, calls to this function NEED to be made inside a condition
        state with the following phrasing:
        if __name__ == "__main__":
        This prevents the code to recursively create instances of this code that would eventually overload the CPUs.

        Returns
        -------
        tuple of Map objects: first element is the map of the calibration peak's FWHM at every point whilst the second is
        the associated uncertainty map.
        """
        data = np.copy(self.data)
        fit_fwhm_list = []
        pool = multiprocessing.Pool()           # This automatically generates an optimal number of workers
        self.reset_update_file()
        start = time.time()
        fit_fwhm_list.append(np.array(pool.map(worker_fit, list((y, data, None, "calibration") for y in range(data.shape[1])))))
        stop = time.time()
        print("Finished in", stop-start, "s.")
        pool.close()
        # The map is temporarily stored in a simple format to facilitate extraction
        fit_fwhm_map = np.squeeze(np.array(fit_fwhm_list), axis=0)
        return (Map(fits.PrimaryHDU(fit_fwhm_map[:,:,0], self.get_header_without_third_dimension())), 
                Map(fits.PrimaryHDU(fit_fwhm_map[:,:,1], self.get_header_without_third_dimension())))

    def fit(self, targeted_ray=int):
        """
        Fit the whole data cube to extract a gaussian's FWHM. This method presupposes that four OH peaks and one Halpha
        peaks are present in the cube's spectrum in addition to the NII peak.
        WARNING: Due to the use of the multiprocessing library, calls to this function NEED to be made inside a condition
        state with the following phrasing:
        if __name__ == "__main__":
        This prevents the code to recursively create instances of this code that would eventually overload the CPUs.

        Arguments
        ---------
        targeted_ray: int. Specifies the ray whose FWHM needs to be extracted. The following legend is used:
        0 : First OH peak
        1 : Second OH peak
        2 : Third OH peak
        3 : Fourth OH peak
        4 : NII peak
        5 : Halpha peak

        Returns
        -------
        tuple of Map objects: first element is the map of the NII peak's FWHM at every point whilst the second is the associated
        uncertainty map.
        """
        data = np.copy(self.data)
        fit_fwhm_list = []
        pool = multiprocessing.Pool()           # This automatically generates an optimal number of workers
        self.reset_update_file()
        start = time.time()
        fit_fwhm_list.append(np.array(pool.map(worker_fit, list((y, data, targeted_ray, "NII") for y in range(data.shape[1])))))
        stop = time.time()
        print("Finished in", stop-start, "s.")
        pool.close()
        # The map is temporarily stored in a simple format to facilitate extraction
        fit_fwhm_map = np.squeeze(np.array(fit_fwhm_list), axis=0)
        return (Map(fits.PrimaryHDU(fit_fwhm_map[:,:,0], self.get_header_without_third_dimension())), 
                Map(fits.PrimaryHDU(fit_fwhm_map[:,:,1], self.get_header_without_third_dimension())))
        
    def bin_cube(self, nb_pix_bin=2):
        """
        Bin a specific cube by the amount of pixels given for every channel.
        Note that this works with every square cube, even though the number of pixels to bin cannot fully divide the cube's size. In
        the case of a rectangular cube, it cannot always find a suitable reshape size.

        Arguments
        ---------
        nb_pix_bin: int, default=2. Specifies the number of pixels to be binned together along a single axis. For example, the default
        value 2 will give a new cube in which every pixel at a specific channel is the mean value of every 4 pixels (2x2 bin) at that
        same channel.

        Returns
        -------
        Data_cube object: binned cube with the same header.
        """
        data = np.copy(self.data)
        # Loop over the nb_pix_bin to find the number of pixels that needs to be cropped
        for i in range(nb_pix_bin):
            try:
                # Create a 5 dimensional array that regroups, for every channel, every group of pixels (2 times the nb_pix_bin)
                # into a new grid whose size has been divided by the number of pixels to bin
                bin_array = data.reshape(data.shape[0], int(data.shape[1]/nb_pix_bin), nb_pix_bin,
                                                        int(data.shape[2]/nb_pix_bin), nb_pix_bin)
                break
            except ValueError:
                # This error occurs if the nb_pix_bin integer cannot fully divide the cube's size
                print(f"Cube to bin will be cut by {i+1} pixel(s).")
                data = data[:,:-1,:-1]
        # The mean value of every pixel group at every channel is calculated and the array returns to a three dimensional state
        return Data_cube(fits.PrimaryHDU(np.nanmean(bin_array, axis=(2,4)), self.bin_header(nb_pix_bin)))

    def get_center_point(self, center_guess=(527, 484)):
        """
        Get the center coordinates of the calibration cube. This method works with a center guess that allows for greater accuracy.
        The center is found by looking at the distances between intensity peaks of the concentric rings.

        Arguments
        ---------
        center_guess: tuple, optional. Defines the point from which intensity maxima will be searched. Any guess close to the center
        may be provided first, but better results will be obtained by re-feeding the function with its own output a few times. The
        arguments' default value is the center coordinates already obtained.

        Returns
        -------
        tuple: the calculated center coordinates with their uncertainty. This is an approximation based on the center_guess provided.
        """
        distances = {"x": [], "y": []}
        # The success int will be used to measure how many distances are utilized to calculate the average center
        success = 0
        for channel in range(1,49):
            # channel_dist = {}
            # The intensity_max dict isolates the x and y axis that pass through the center_guess
            intensity_max = {
                "intensity_x": self.data[channel-1, center_guess[1], :],
                "intensity_y": self.data[channel-1, :, center_guess[0]]
            }
            for name, axis_list in intensity_max.items():
                # The axes_pos list collects the detected intensity peaks along each axis
                axes_pos = []
                for coord in range(1, 1023):
                    if (axis_list[coord-1] < axis_list[coord] > axis_list[coord+1]
                         and axis_list[coord] > 500):
                        axes_pos.append(coord)
                # Here the peaks are verified to make sure they correspond to the max intensity of each ring
                for i in range(len(axes_pos)-1):
                    # If two peaks are too close to each other, they cannot both be real peaks
                    if axes_pos[i+1] - axes_pos[i] < 50:
                        if axis_list[axes_pos[i]] > axis_list[axes_pos[i+1]]:
                            axes_pos[i+1] = 0
                        else:
                            axes_pos[i] = 0
                # The fake peaks are removed from the list
                axes_pos = np.setdiff1d(axes_pos, 0)
                
                # Here is only considered the case where the peaks of two concentric rings have been detected in the two directions
                if len(axes_pos) == 4:
                    dists = [(axes_pos[3] - axes_pos[0])/2 + axes_pos[0], (axes_pos[2] - axes_pos[1])/2 + axes_pos[1]]
                    distances[name[-1]].append(dists)
                    success += 1
        x_mean, y_mean = np.mean(distances["x"], axis=(0,1)), np.mean(distances["y"], axis=(0,1))
        x_uncertainty, y_uncertainty = np.std(distances["x"], axis=(0,1)), np.std(distances["y"], axis=(0,1))
        return [x_mean, x_uncertainty], [y_mean, y_uncertainty]
    
    def get_header_without_third_dimension(self):
        """
        Get the adaptation of a Data_cube object's header for a Map object by removing the spectral axis.

        Returns
        -------
        astropy.io.fits.header.Header: header with the same but with the third axis removed.
        """
        header = self.header.copy()
        wcs = WCS(header)
        wcs.sip = None
        wcs = wcs.dropaxis(2)
        header = wcs.to_header(relax=True)
        return header


def worker_fit(args):
    """
    Fit an entire line of a Data_cube.

    Arguments
    ---------
    args: tuple. The first element is the y value of the line to be fitted, the second element is the Data_cube used and the
    third element is a string specifying if the cube is a calibration cube or a NII cube: "calibration" or "NII".
    Note that arguments are given in tuple due to the way the multiprocessing library operates.

    Returns
    -------
    list: FWHM value of the fitted gaussian on the studied peak along the specified line. Each coordinates has two values:
    the former being the FWHM value whilst the latter being its uncertainty.
    """
    y, data, targeted_ray, cube_type = args
    line = []
    if cube_type == "calibration":
        for x in range(data.shape[2]):
            spectrum_object = Spectrum(data[:,y,x], calibration=True)
            spectrum_object.fit_calibration()
            try:
                line.append(spectrum_object.get_FWHM_speed(
                            spectrum_object.get_fitted_gaussian_parameters(), spectrum_object.get_uncertainties()["g0"]["stddev"]))
            except:
                print(x,y)
                line.append([np.NAN, np.NAN])
        Data_cube.give_update(None, f"Calibration fitting progress /{data.shape[2]}")

    elif cube_type == "NII":
        for x in range(data.shape[2]):
            spectrum_object = Spectrum(data[:,y,x], calibration=False)
            spectrum_object.fit_data_cube(spectrum_object.get_initial_guesses())
            line.append(spectrum_object.get_FWHM_speed(
                        spectrum_object.get_fitted_gaussian_parameters()[targeted_ray],
                        spectrum_object.get_uncertainties()[f"g{targeted_ray}"]["stddev"]))
        Data_cube.give_update(None, f"NII fitting progress /{data.shape[2]}")
    return line



class Map(Fits_file):
    """
    Encapsulate the necessary methods to compare and treat maps.
    """

    def __init__(self, fits_object):
        """
        Initialize a Map object.

        Arguments
        ---------
        fits_object: astropy.io.fits.hdu.image.PrimaryHDU. Contains the data values and header of the map.
        """
        self.object = fits_object
        self.data = fits_object.data
        self.header = fits_object.header

    def __add__(self, other):
        assert self.data.shape == other.data.shape, "Maps of different sizes are being added."
        return Map(fits.PrimaryHDU(self.data + other.data, self.header))
    
    def __sub__(self, other):
        assert self.data.shape == other.data.shape, "Maps of different sizes are being subtracted."
        return Map(fits.PrimaryHDU(self.data - other.data, self.header))

    def __pow__(self, power):
        return Map(fits.PrimaryHDU(self.data ** power, self.header))
    
    def __array__(self):
        return self.data
    
    def __eq__(self, other):
        return np.array_equal(self.data, other.data)

    def plot_map(self, bounds=None):
        """
        Plot the map in matplotlib.pyplot.

        Arguments
        ---------
        bounds: tuple, optional. Indicates the colorbar's bounds if an autoscale is not desired. The tuple's first element is
        the minimum and the second is the maximum.
        """
        data = np.copy(self.data)
        if bounds:
            plt.colorbar(plt.imshow(data, origin="lower", cmap="viridis", vmin=bounds[0], vmax=bounds[1]))
        else:
            plt.colorbar(plt.imshow(data, origin="lower", cmap="viridis"))
        plt.show()

    def plot_two_maps(self, other, bounds=None, alpha=0.5):
        """
        Plot two maps superposed with a certain alpha. The first map is plotted with the viridis colormap whereas the second
        map is plotted with the magma colormap.

        Arguments
        ---------
        other: Map object. The second map that will be plotted with the magma colormap.
        bounds: tuple, optional. Indicates the colorbar's bounds if an autoscale is not desired. The tuple's first element is
        the minimum and the second is the maximum.
        """
        plt.colorbar(plt.imshow(self.data, origin="lower", cmap="viridis", vmin=bounds[0], vmax=bounds[1], alpha=alpha))
        plt.colorbar(plt.imshow(other.data, origin="lower", cmap="magma", vmin=bounds[0], vmax=bounds[1], alpha=alpha))
        plt.show()

    def bin_map(self, nb_pix_bin=2):
        """
        Bin the map by the amount of pixels given.
        Note that this works with every square map, even though the number of pixels to bin cannot fully divide the map's size. In
        the case of a rectangular map, it cannot always find a suitable reshape size.

        Arguments
        ---------
        nb_pix_bin: int. Specifies the number of pixels to be binned together along a single axis. For example, the default value 2
        will give a new map in which every pixel is the mean value of every 4 pixels (2x2 bin).

        Returns
        -------
        Map object: binned map.
        """
        data = np.copy(self.data)
        # Loop over the nb_pix_bin to find the number of pixels that needs to be cropped
        for i in range(nb_pix_bin):
            try:
                # Create a 4 dimensional array that regroups every group of pixels (2 times the nb_pix_bin) into a new grid whose
                # size has been divided by the number of pixels to bin
                bin_array = data.reshape(int(data.shape[0]/nb_pix_bin), nb_pix_bin, int(data.shape[1]/nb_pix_bin), nb_pix_bin)
                break
            except ValueError:
                # This error occurs if the nb_pix_bin integer cannot fully divide the map's size
                print(f"Map to bin will be cut by {i+1} pixel(s).")
                data = data[:-1,:-1]
        # The mean value of every pixel group is calculated and the array returns to a two dimensional state
        return Map(fits.PrimaryHDU(np.nanmean(bin_array, axis=(1,3)), self.bin_header(nb_pix_bin)))
    
    def smooth_order_change(self, uncertainty_map, center=(527, 484)):
        """
        Smooth the fitted FWHM of the calibration cube for the first two interference order changes. This is needed as the FWHM
        is reduced at points where the calibration peak changes of interference order. This changes the pixels' value in an
        order change to the mean value of certain pixels in a 7x7 area around the pixel.

        Arguments
        ---------
        uncertainty_map: Map object. Map of the FWHM's uncertainty of the fitted peak at every point.
        center: tuple. Specifies the coordinates of the interference pattern's center pixel.

        Returns
        -------
        tuple of Map objects: the first Map is the FWHM at every point and the second is its associated uncertainty.
        """
        data = np.copy(self.data)
        uncertainties = np.copy(uncertainty_map.data)
        center = round(center[0]), round(center[1])
        # The bin_factor corrects the distances in the case of a binned array
        bin_factor = center[0] / 527
        # The smoothing_max_thresholds list of ints is defined by trial and error and tunes the pixels to calculate the mean
        # The first element is used for the first interference order change and the second element is for the second change
        smoothing_max_thresholds = [0.4, 1.8]
        # The bounds list define the area in which the FWHM minimum will be searched, corresponding to an order change
        bounds = [
            np.array((255,355)) * bin_factor,
            np.array((70,170)) * bin_factor
        ]
        # The peak_regions list gives the data's values within the bounds for the two regions
        peak_regions = [
            list(data[center[1], int(bounds[0][0]):int(bounds[0][1])]),
            list(data[center[1], int(bounds[1][0]):int(bounds[1][1])])
        ]
        # The radiuses list gives the position of the minimum fwhms relative to the center of the image
        radiuses = [
            center[0] - (peak_regions[0].index(min(peak_regions[0])) + bounds[0][0]),
            center[0] - (peak_regions[1].index(min(peak_regions[1])) + bounds[1][0])
        ]

        for x in range(data.shape[1]):
            for y in range(data.shape[0]):
                current_radius = np.sqrt((x-center[0])**2 + (y-center[1])**2)

                # The 5 and 4 ints have been observed to offer better results considering the rings' width
                if (radiuses[0] - 5*bin_factor <= current_radius <= radiuses[0] + 5*bin_factor or
                    radiuses[1] - 4*bin_factor <= current_radius <= radiuses[1] + 4*bin_factor):
                    near_pixels = np.copy(data[y-3:y+4, x-3:x+4])
                    near_pixels_uncertainty = np.copy(uncertainties[y-3:y+4, x-3:x+4])

                    if radiuses[0] - 4*bin_factor <= current_radius <= radiuses[0] + 4*bin_factor:
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[0]] = np.NAN
                    else:
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[1]] = np.NAN
                    
                    data[y,x] = np.nanmean(near_pixels)
                    uncertainties[y,x] = np.nanmean(near_pixels * 0 + near_pixels_uncertainty)
                    # The addition of near_pixels * 0 makes it so the pixels that have np.NAN will not be used
        return Map(fits.PrimaryHDU(data, self.header)), Map(fits.PrimaryHDU(uncertainties, self.header))

    def reproject_on(self, other):
        """
        Get the reprojection of the map on the other object's WCS. This makes the coordinates match.

        Arguments
        ---------
        other: Map object. Reference map to project on and to base the shift of WCS.

        Returns
        -------
        Map object: map with WCS aligned to the other map.
        """
        reprojection = reproject_interp(self.object, other.header, return_footprint=False, order="nearest-neighbor")
        return Map(fits.PrimaryHDU(reprojection, other.header))
    
    def align_regions(self, uncertainty_map):
        """
        Get the squared FWHM map in which the instrumental function has been subtracted with the three regions corrected to fit
        better the WCS. The same method can also be called with an uncertainty map to get the aligned uncertainty map.

        Arguments
        ---------
        uncertainty_map: Map object. Gives the FWHM's uncertainty at every point.

        Returns
        -------
        tuple of Map objects: first element is the global map with the three aligned regions, result of the subtraction of the
        squared FWHM map and the squared instrumental function map and the second element is the uncertainty.
        """
        regions = [
            pyregion.open("gaussian_fitting/regions/region_1.reg"),
            pyregion.open("gaussian_fitting/regions/region_2.reg"),
            pyregion.open("gaussian_fitting/regions/region_3.reg")
        ]

        # A mask of zeros and ones is created with the regions
        masks = [region.get_mask(hdu=self.object) for region in regions]
        masks = [np.where(mask == False, 0, mask) for mask in masks]
        masks = [np.where(mask == True, 1, mask) for mask in masks]

        # The map's data is removed where a mask applies
        new_data = np.copy(self.data) * (1 - (masks[0] + masks[1] + masks[2]))
        # Every specific map has the same values than the global map, but the header is changed to fit a specific region
        specific_maps = [
            Map(fits.open(f"gaussian_fitting/maps/reproject/region_1_widening.fits")[0]),
            Map(fits.open(f"gaussian_fitting/maps/reproject/region_2_widening.fits")[0]),
            Map(fits.open(f"gaussian_fitting/maps/reproject/region_3_widening.fits")[0])
        ]
        # The calibration map's correction is applied
        calib_map = Map(fits.open(f"gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")[0]).bin_map(2)
        modified_specific_maps = [specific_map**2 - calib_map**2 for specific_map in specific_maps]
        # Alignment of the specific maps on the global WCS
        modified_specific_maps = [specific_map.reproject_on(self) for specific_map in modified_specific_maps]
        # Only the data within the mask is kept
        region_data = [specific_map.data * masks[i] for i, specific_map in enumerate(modified_specific_maps)]
        new_data += region_data[0] + region_data[1] + region_data[2]

        # The uncertainty map's data is removed where a mask applies
        new_data_unc = np.copy(uncertainty_map.data) * (1 - (masks[0] + masks[1] + masks[2]))
        specific_maps_unc = [
            Map(fits.open(f"gaussian_fitting/maps/reproject/region_1_widening_unc.fits")[0]),
            Map(fits.open(f"gaussian_fitting/maps/reproject/region_2_widening_unc.fits")[0]),
            Map(fits.open(f"gaussian_fitting/maps/reproject/region_3_widening_unc.fits")[0])
        ]
        # The calibration map's uncertainty is propagated with each specific_map and specific_map_unc
        calib_map_unc = Map(fits.open(f"gaussian_fitting/maps/computed_data/smoothed_instr_f_unc.fits")[0]).bin_map(2)
        modified_specific_maps_unc = [maps[0].calc_power_uncertainty(maps[1], 2) + calib_map.calc_power_uncertainty(calib_map_unc, 2)
                                      for maps in list(zip(specific_maps, specific_maps_unc))]
        # Alignment of the specific maps uncertainty on the main map's WCS
        modified_specific_maps_unc = [specific_map_unc.reproject_on(self) for specific_map_unc in modified_specific_maps_unc]
        # Only the data within the mask is kept
        region_data_unc = [specific_map_unc.data * masks[i] for i, specific_map_unc in enumerate(modified_specific_maps_unc)]
        new_data_unc += region_data_unc[0] + region_data_unc[1] + region_data_unc[2]
        return Map(fits.PrimaryHDU(new_data, self.header)), Map(fits.PrimaryHDU(new_data_unc, uncertainty_map.header))

    def calc_power_uncertainty(self, uncertainty_map, power):
        """
        Get the propagated uncertainty of a quantity raised to any power using uncertainty propagation rules.

        Arguments
        ---------
        uncertainty_map: map of the uncertainty associated to the self.data.
        power: power at which the data is raised.
        
        Returns
        -------
        Map object: map of the correct uncertainty following the exponential operation. The header of the object is the
        uncertainty_map's header.
        """
        return Map(fits.PrimaryHDU((uncertainty_map.data / self.data * power * self.data**power), uncertainty_map.header))

    def transfer_temperature_to_FWHM(self):
        """
        Get the FWHM of the thermal Doppler broadening. This is used to convert the temperature map into a FWHM map that
        can be compared with other FWHM maps. This function can be called with an uncertainty Map object to get the FWHM's
        uncertainty.

        Returns
        -------
        Map object: map of the FWHM due to thermal Doppler broadening.
        """
        angstroms_center = 6583.41              # Emission wavelength of NII 
        m = 14.0067 * scipy.constants.u         # Nitrogen mass
        c = scipy.constants.c                   # Light speed
        k = scipy.constants.k                   # Boltzmann constant
        angstroms_FWHM = 2 * np.sqrt(2 * np.log(2)) * angstroms_center * np.sqrt(self.data * k / (c**2 * m))
        speed_FWHM = c * angstroms_FWHM / angstroms_center / 1000
        return Map(fits.PrimaryHDU(speed_FWHM, self.header))
