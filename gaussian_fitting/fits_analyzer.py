from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
import scipy

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from reproject import reproject_interp
import pyregion

from cube_spectrum import Spectrum

import multiprocessing
import time
import os

if not sys.warnoptions:
    warnings.simplefilter("ignore")



class Fits_file():
    """
    Encapsulate the methods specific to .fits files that can be used both in data cube and map analysis.
    """

    def bin_header(self, nb_pix_bin: int=2) -> fits.header:
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
    
    def save_as_fits_file(self, filename: str):
        """
        Write an array as a fits file of the specified name with or without a header. If the object has a header, it will be saved.

        Arguments
        ---------
        filename: str. Indicates the path and name of the created file. If the file already exists, a warning will appear and the
        file can be overwritten.
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

    def give_update(self, info: str):
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

    def __init__(self, fits_object: fits.PrimaryHDU):
        """
        Initialize an analyzer object. The datacube's file name must be given.

        Arguments
        ---------
        fits_object: astropy.io.fits.hdu.image.PrimaryHDU. Contains the data values and header of the data cube.
        """
        self.object = fits_object
        self.data = fits_object.data
        self.header = fits_object.header
        # self.data = self.data[:,260:270,260:270]

    def fit_calibration(self) -> Map_u:
        """
        Fit the whole data cube as if it was a calibration cube and extract the FWHM and its uncertainty at every point.
        WARNING: Due to the use of the multiprocessing library, calls to this function NEED to be made inside a condition
        state with the following phrasing:
        if __name__ == "__main__":
        This prevents the code to recursively create instances of itself that would eventually overload the CPUs.

        Returns
        -------
        Map_u object: map of the FWHM value and its associated uncertainty.
        """
        data = np.copy(self.data)
        fit_fwhm_list = []
        pool = multiprocessing.Pool()           # This automatically generates an optimal number of workers
        self.reset_update_file()
        start = time.time()
        fit_fwhm_list.append(np.array(pool.map(worker_fit, list((y, data, "calibration") for y in range(data.shape[1])))))
        stop = time.time()
        print("Finished in", stop-start, "s.")
        pool.close()
        # The map is temporarily stored in a simple format to facilitate extraction
        fit_fwhm_map = np.squeeze(np.array(fit_fwhm_list), axis=0)
        return Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0], self.get_header_without_third_dimension()),
                                   fits.ImageHDU(fit_fwhm_map[:,:,1], None)]))

    def fit(self) -> Maps:
        """
        Fit the whole data cube to extract a gaussian's FWHM. This method presupposes that four OH peaks and one Halpha
        peak are present in the cube's spectrum in addition to the NII peak.
        WARNING: Due to the use of the multiprocessing library, calls to this function NEED to be made inside a condition
        state with the following phrasing:
        if __name__ == "__main__":
        This prevents the code to recursively create instances of itself that would eventually overload the CPUs.

        Arguments
        ---------
        targeted_ray: str. Specifies the ray whose FWHM needs to be extracted. The supported peaks are:
        "OH1", "OH2", "OH3", "OH4", "NII" and "Ha".

        Returns
        -------
        Maps object: maps of every ray's FWHM present in the provided data cube. Note that each map is Map_usnr object.
        """
        data = np.copy(self.data)
        fit_fwhm_list = []
        # cube_type = "NII"
        # if calculate_snr:
        #     cube_type = "NII with snr"
        pool = multiprocessing.Pool()           # This automatically generates an optimal number of workers
        self.reset_update_file()
        start = time.time()
        fit_fwhm_list.append(np.array(pool.map(worker_fit, list((y, data, "NII") for y in range(data.shape[1]))))) # calculate_snr & cube_type removed
        stop = time.time()
        print("Finished in", stop-start, "s.")
        pool.close()
        new_header = self.get_header_without_third_dimension()
        fit_fwhm_array = np.squeeze(np.array(fit_fwhm_list))
        map_list = Maps([
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,0,0], new_header),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,1]),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,2])]), name="OH1_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,1,0], new_header),
                                   fits.ImageHDU(fit_fwhm_array[:,:,1,1]),
                                   fits.ImageHDU(fit_fwhm_array[:,:,1,2])]), name="OH2_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,2,0], new_header),
                                   fits.ImageHDU(fit_fwhm_array[:,:,2,1]),
                                   fits.ImageHDU(fit_fwhm_array[:,:,2,2])]), name="OH3_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,3,0], new_header),
                                   fits.ImageHDU(fit_fwhm_array[:,:,3,1]),
                                   fits.ImageHDU(fit_fwhm_array[:,:,3,2])]), name="OH4_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,4,0], new_header),
                                   fits.ImageHDU(fit_fwhm_array[:,:,4,1]),
                                   fits.ImageHDU(fit_fwhm_array[:,:,4,2])]), name="NII_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,5,0], new_header),
                                   fits.ImageHDU(fit_fwhm_array[:,:,5,1]),
                                   fits.ImageHDU(fit_fwhm_array[:,:,5,2])]), name="Ha_fwhm"),
            Map(fits.PrimaryHDU(fit_fwhm_array[:,:,6,0], new_header), name="7_component_fit")
        ])
        return map_list
        # The map is temporarily stored in a simple format to facilitate extraction
        # fit_fwhm_map = np.squeeze(np.array(fit_fwhm_list), axis=0)
        # if calculate_snr:
        #     return Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0], new_header),
        #                                fits.ImageHDU(fit_fwhm_map[:,:,1], new_header),
        #                                fits.ImageHDU(fit_fwhm_map[:,:,2], new_header)]))
        # return Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0], new_header),
        #                            fits.ImageHDU(fit_fwhm_map[:,:,1], new_header)]))
        
    def bin_cube(self, nb_pix_bin: int=2) -> Data_cube:
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

    def get_center_point(self, center_guess: tuple=(527, 484)) -> tuple:
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
    
    def get_header_without_third_dimension(self) -> fits.header:
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


def worker_fit(args: tuple) -> list:
    """
    Fit an entire line of a Data_cube.

    Arguments
    ---------
    args: tuple. The first element is the y value of the line to be fitted, the second element is the Data_cube used, the third
    element is the ray that needs to be extracted in the case of a multi-gaussian fit and the fourth element is a string
    specifying if the cube is a calibration cube or a NII cube: "calibration" or "NII".
    Note that arguments are given in tuple due to the way the multiprocessing library operates.

    Returns
    -------
    list: FWHM value of the fitted gaussian on the studied peak along the specified line. Each coordinates has two values:
    the former being the FWHM value whilst the latter being its uncertainty.
    """
    y, data, cube_type = args
    line = []
    if cube_type == "calibration":
        for x in range(data.shape[2]):
            spectrum_object = Spectrum(data[:,y,x], calibration=True)
            spectrum_object.fit_calibration()
            try:
                line.append(spectrum_object.get_FWHM_speed())
            except:
                line.append([np.NAN, np.NAN])
        Data_cube.give_update(None, f"Calibration fitting progress /{data.shape[2]}")

    elif cube_type == "NII":
        for x in range(data.shape[2]):
            spectrum_object = Spectrum(data[:,y,x], calibration=False)
            spectrum_object.fit_NII_cube()
            a = np.array([spectrum_object.get_snr("OH1")])
            line.append(np.array((
                np.concatenate((spectrum_object.get_FWHM_speed("OH1"), np.array([spectrum_object.get_snr("OH1")]))),
                np.concatenate((spectrum_object.get_FWHM_speed("OH2"), np.array([spectrum_object.get_snr("OH2")]))),
                np.concatenate((spectrum_object.get_FWHM_speed("OH3"), np.array([spectrum_object.get_snr("OH3")]))),
                np.concatenate((spectrum_object.get_FWHM_speed("OH4"), np.array([spectrum_object.get_snr("OH4")]))),
                np.concatenate((spectrum_object.get_FWHM_speed("NII"), np.array([spectrum_object.get_snr("NII")]))),
                np.concatenate((spectrum_object.get_FWHM_speed("Ha"), np.array([spectrum_object.get_snr("Ha")]))),
                np.array([spectrum_object.seven_components_fit, False, False])
            )))
        Data_cube.give_update(None, f"NII complete cube fitting progress /{data.shape[2]}")


    elif cube_type == "NII":
        for x in range(data.shape[2]):
            spectrum_object = Spectrum(data[:,y,x], calibration=False)
            spectrum_object.fit_NII_cube()
            line.append(spectrum_object.get_FWHM_speed(targeted_ray))
        Data_cube.give_update(None, f"NII fitting progress /{data.shape[2]}")
    
    elif cube_type == "NII with snr":
        for x in range(data.shape[2]):
            spectrum_object = Spectrum(data[:,y,x], calibration=False)
            spectrum_object.fit_NII_cube()
            fwhm_values = spectrum_object.get_FWHM_speed(targeted_ray)
            line.append(np.concatenate((fwhm_values, np.array([(spectrum_object.get_fitted_gaussian_parameters(targeted_ray).amplitude
                                                                /u.Jy)/spectrum_object.get_residue_stddev()]))))
        Data_cube.give_update(None, f"NII with snr fitting progress /{data.shape[2]}")
    return line



class Map(Fits_file):
    """
    Encapsulate the necessary methods to compare and treat maps.
    """

    def __init__(self, fits_object, name: str=None):
        """
        Initialize a Map object.

        Arguments
        ---------
        fits_object: astropy.io.fits.hdu.image.PrimaryHDU. Contains the data values and header of the map.
        name: str, default=None. Name of the Map. This is primarily used with the Maps object.
        """
        self.object = fits_object
        self.data = fits_object.data
        self.header = fits_object.header
        if name is not None:
            self.name = name

    def __add__(self, other):
        if type(other) == Map:
            assert self.data.shape == other.data.shape, "Maps of different sizes are being added."
            return Map(fits.PrimaryHDU(self.data + other.data, self.header))
        else:
            return Map(fits.PrimaryHDU(self.data + other, self.header))
    
    def __sub__(self, other):
        assert self.data.shape == other.data.shape, "Maps of different sizes are being subtracted."
        return Map(fits.PrimaryHDU(self.data - other.data, self.header))

    def __pow__(self, power):
        return Map(fits.PrimaryHDU(self.data ** power, self.header))
    
    def __mul__(self, other):
        return Map(fits.PrimaryHDU(self.data * other, self.header))
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        return Map(fits.PrimaryHDU(self.data / other, self.header))

    def __rtruediv__(self, other):
        return (self.__truediv__(other))**(-1)

    def __array__(self):
        return self.data
    
    def __eq__(self, other):
        return np.nansum(self.data - other.data) < 10**(-10)

    def copy(self):
        return Map(fits.PrimaryHDU(np.copy(self.data), self.header.copy()))

    def plot_map(self, bounds: tuple=None):
        """
        Plot the map in matplotlib.pyplot.

        Arguments
        ---------
        bounds: tuple, optional. Indicates the colorbar's bounds if an autoscale is not desired. The tuple's first element is
        the minimum and the second is the maximum.
        """
        if bounds:
            plt.colorbar(plt.imshow(self.data, origin="lower", cmap="viridis", vmin=bounds[0], vmax=bounds[1]))
        else:
            plt.colorbar(plt.imshow(self.data, origin="lower", cmap="viridis"))
        plt.show()

    def plot_two_maps(self, other: Map, bounds: tuple=None):
        """
        Plot two maps superposed with a certain alpha. The first map is plotted with the viridis colormap whereas the second
        map is plotted with the magma colormap.

        Arguments
        ---------
        other: Map object. The second map that will be plotted with the magma colormap.
        bounds: tuple, optional. Indicates the colorbar's bounds if an autoscale is not desired. The tuple's first element is
        the minimum and the second is the maximum.
        """
        if bounds is None:
            ax1 = plt.subplot(1,2,1)
            ax2 = plt.subplot(1,2,2)
            plt.colorbar(ax1.imshow(self.data, origin="lower", cmap="viridis"))
            plt.colorbar(ax2.imshow(other.data, origin="lower", cmap="viridis"))
        else:
            ax1 = plt.subplot(1,2,1)
            ax2 = plt.subplot(1,2,2)
            plt.colorbar(ax1.imshow(self.data, origin="lower", cmap="viridis", vmin=bounds[0], vmax=bounds[1]))
            plt.colorbar(ax2.imshow(other.data, origin="lower", cmap="viridis", vmin=bounds[0], vmax=bounds[1]))
        plt.show()

    def bin_map(self, nb_pix_bin: int=2, raw_data: np.ndarray=None) -> Map:
        """
        Bin the map by the amount of pixels given.
        Note that this works with every square map, even though the number of pixels to bin cannot fully divide the map's size. In
        the case of a rectangular map, it cannot always find a suitable reshape size.

        Arguments
        ---------
        nb_pix_bin: int. Specifies the number of pixels to be binned together along a single axis. For example, the default value 2
        will give a new map in which every pixel is the mean value of every 4 pixels (2x2 bin).
        raw_data: numpy array, default=None. If present, specifies the data to bin. By default, the Map's data will be binned.

        Returns
        -------
        Map object: binned map.
        """
        if raw_data is None:
            data = np.copy(self.data)
        else:
            data = np.copy(raw_data)
        
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
    
    def smooth_order_change(self, center: tuple=(527, 484)) -> Map:
        """
        Smooth the fitted FWHM of the calibration cube for the first two interference order changes. This is needed as the FWHM
        is reduced at points where the calibration peak changes of interference order. This changes the pixels' value in an
        order change to the mean value of certain pixels in a 7x7 area around the pixel.

        Arguments
        ---------
        center: tuple of ints. Specifies the coordinates of the interference pattern's center pixel.

        Returns
        -------
        Map object: FWHM at every point.
        """
        data = np.copy(self.data)
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
                    near_pixels = np.copy(self.data[y-3:y+4, x-3:x+4])

                    if radiuses[0] - 4*bin_factor <= current_radius <= radiuses[0] + 4*bin_factor:
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[0]] = np.NAN
                    else:
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[1]] = np.NAN
                    
                    data[y,x] = np.nanmean(near_pixels)
                    # The addition of near_pixels * 0 makes it so the pixels that have np.NAN will not be used
        return Map(fits.PrimaryHDU(data, self.header))

    def reproject_on(self, other: Map) -> Map:
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
    
    def align_regions(self) -> Map:
        """
        Get the squared FWHM map in which the instrumental function has been subtracted with the three regions corrected to fit
        better the WCS.

        Returns
        -------
        Map object: global map with the three aligned regions, result of the subtraction of the squared FWHM map and the squared
        instrumental function map.
        """
        regions = [
            pyregion.open("gaussian_fitting/regions/region_1.reg"),
            pyregion.open("gaussian_fitting/regions/region_2.reg"),
            pyregion.open("gaussian_fitting/regions/region_3.reg")
        ]

        # A mask of zeros and ones is created with the regions
        masks = [region.get_mask(hdu=self.object) for region in regions]
        masks = [np.where(mask == False, 0, 1) for mask in masks]

        # The map's data is removed where a mask applies
        new_map = self.copy() * (1 - (masks[0] + masks[1] + masks[2]))
        # Every specific map needs to have the same values than the global map, but the header is changed to fit a specific region
        # The right headers are first opened, then the values are changed
        specific_headers = [
            Map(fits.open(f"gaussian_fitting/maps/reproject/region_1_widening.fits")[0]).header,
            Map(fits.open(f"gaussian_fitting/maps/reproject/region_2_widening.fits")[0]).header,
            Map(fits.open(f"gaussian_fitting/maps/reproject/region_3_widening.fits")[0]).header
        ]
        # The real data is inserted
        specific_maps = []
        for header in specific_headers:
            specific_maps.append(Map(fits.PrimaryHDU(np.copy(self.data), header)))
        # Alignment of the specific maps on the global WCS
        reprojected_specific_maps = [specific_map.reproject_on(self) for specific_map in specific_maps]
        # Only the data within the mask is kept
        region_data = [specific_map * masks[i] for i, specific_map in enumerate(reprojected_specific_maps)]
        new_map += region_data[0] + region_data[1] + region_data[2]
        return new_map

    def transfer_temperature_to_FWHM(self) -> Map:
        """
        Get the FWHM of the thermal Doppler broadening. This is used to convert the temperature map into a FWHM map that
        can be compared with other FWHM maps. This method uses the NII peak's wavelength for the Doppler calculations.

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
    
    def get_region_statistics(self, region: pyregion.core.ShapeList, plot_histogram: bool=False) -> dict:
        """
        Get the statistics of a region along with a histogram. The supported statistic measures are: median, mean, standard
        deviation, skewness and kurtosis.

        Arguments
        ---------
        region: pyregion.core.ShapeList. Region in which the statistics need to be calculated. A histogram will also be made with
        the data in this region.
        plot_histogram: bool, default=False. Boolean that specifies if the histogram should be plotted.

        Returns
        -------
        dict: statistics of the region. Every key is a statistic measure.
        """
        # A mask of zeros and ones is created with the region
        try:
            mask = region.get_mask(hdu=self.object)
        except:
            mask = region.get_mask(hdu=self.object[0])
        mask = np.where(mask == False, np.nan, 1)
        # The map's data is only kept where a mask applies
        new_map = self.copy() * mask
        stats = {
            "median": np.nanmedian(new_map.data),
            "mean": np.nanmean(new_map.data),
            "standard_deviation": np.nanstd(new_map.data),
            "skewness": scipy.stats.skew(new_map.data, axis=None, nan_policy="omit"),
            "kurtosis": scipy.stats.kurtosis(new_map.data, axis=None, nan_policy="omit")
        }
        # The NANs are removed from the data from which the statistics are computed
        map_data_without_nan = np.ma.masked_invalid(new_map.data).compressed()
        if plot_histogram:
            plt.hist(map_data_without_nan, bins=np.histogram_bin_edges(map_data_without_nan, bins="fd"))
            
        return stats



class Map_u(Map):
    """
    Encapsulate the methods specific to maps with uncertainties.
    Note that a Map_u is essentially two Map objects into a single object, the first Map being the data and the second one
    being its uncertainty. This makes conversion from Map_u -> Map easier via one of the following statements:
    data_map, uncertainty_map = Map_u | uncertainty_map = Map_u[1]
    data_map and uncertainty_map would then be two Map objects.
    It is also possible to create a Map_u object from two Map objects using the from_map_objects method.
    """

    def __init__(self, fits_list, name: str=None):
        """
        Initialize a Map_u object. 

        Arguments
        ---------
        fits_list: astropy.io.fits.hdu.hdulist.HDUList. List of astropy objects. Contains the values, uncertainties and header
        of the map.
        name: str, default=None. Name of the Map_u. This is primarily used with the Maps object.
        """
        self.object = fits_list
        self.data = fits_list[0].data
        self.uncertainties = fits_list[1].data
        self.header = fits_list[0].header
        assert self.data.shape == self.uncertainties.shape, "The data and uncertainties sizes do not match."
        if name is not None:
            self.name = name
    
    @classmethod
    def from_Map_objects(self, map_data: Map, map_uncertainty: Map) -> Map_u:
        """
        Create a Map_u object using two Map objects. An object may be created using the following statement:
        new_map = Map_u.from_Map_objects(data_map, uncertainties_map),
        where data_map and uncertainties_map are two Map objects.

        Arguments
        ---------
        map_data: Map object. Serves as the Map_u's data.
        map_uncertainty: Map object. Serves as the Map_u's uncertainties.

        Returns
        -------
        Map_u object: map with the corresponding data and uncertainties.
        """
        return self(fits.HDUList([fits.PrimaryHDU(map_data.data, map_data.header),
                                  fits.ImageHDU(map_uncertainty.data, map_data.header)]))

    def __add__(self, other):
        if type(other) == Map_u:
            assert self.data.shape == other.data.shape, "Maps of different sizes are being added."
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data + other.data, self.header),
                                       fits.ImageHDU(self.uncertainties + other.uncertainties, self.header)]))
        else:
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data + other, self.header),
                                       fits.ImageHDU(self.uncertainties, self.header)]))
    
    def __sub__(self, other):
        if type(other) == Map_u:
            assert self.data.shape == other.data.shape, "Maps of different sizes are being subtracted."
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data - other.data, self.header),
                                       fits.ImageHDU(self.uncertainties + other.uncertainties, self.header)]))
        else:
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data - other, self.header),
                                       fits.ImageHDU(self.uncertainties, self.header)]))

    def __pow__(self, power):
        return Map_u(fits.HDUList([fits.PrimaryHDU(self.data ** power, self.header),
                                   fits.ImageHDU(np.abs(self.uncertainties / self.data * power * self.data**power), self.header)]))
    
    def __mul__(self, other):
        return Map_u(fits.HDUList([fits.PrimaryHDU(self.data * other, self.header),
                                   fits.ImageHDU(self.uncertainties * other, self.header)]))
        
    def __truediv__(self, other):
        return Map_u(fits.HDUList([fits.PrimaryHDU(self.data / other, self.header),
                                   fits.ImageHDU(self.uncertainties / other, self.header)]))

    def __eq__(self, other):
        return (np.nansum(self.data - other.data) == 0. and 
                np.nansum(self.uncertainties - other.uncertainties) == 0.)
    
    def copy(self):
        return Map_u(fits.HDUList([fits.PrimaryHDU(np.copy(self.data), self.header.copy()),
                                   fits.ImageHDU(np.copy(self.uncertainties), self.header.copy())]))
    
    def __iter__(self):
        self.n = -1
        return self
    
    def __next__(self):
        self.n += 1
        if self.n > len(self.object) - 1:
            raise StopIteration
        else:
            return Map(fits.PrimaryHDU(self.object[self.n].data, self.header))
        
    def __getitem__(self, index):
        return Map(self.object[index])

    def save_as_fits_file(self, filename: str):
        """
        Write the Map_u as a fits file of the specified name with or without a header. If the object has a header, it will be
        saved. The data uncertainty is saved as the [1] extension of the fits file. To view the uncertainty map on DS9, simply
        open the file with the following path: File -> Open as -> Multi Extension Cube. The data and its uncertainty will then
        be visible just like in a data cube.
        
        Arguments
        ---------
        filename: str. Indicates the path and name of the created file. If the file already exists, a warning will appear and the
        file can be overwritten.
        """
        # Check if the file already exists
        try:
            fits.open(filename)[0]
            # The file already exists
            while True:
                answer = input(f"The file '{filename}' already exists, do you wish to overwrite it ? [y/n]")
                if answer == "y":
                    hdu_list = fits.HDUList([
                        fits.PrimaryHDU(self.data, self.header),
                        fits.ImageHDU(self.uncertainties, self.header)
                    ])

                    hdu_list.writeto(filename, overwrite=True)
                    print("File overwritten.")
                    break

                elif answer == "n":
                    break
                
        except:
            # The file does not yet exist
            hdu_list = fits.HDUList([
                fits.PrimaryHDU(self.data, self.header),
                fits.ImageHDU(self.uncertainties, self.header)
            ])

            hdu_list.writeto(filename, overwrite=True)

    def bin_map(self, nb_pix_bin: int=2) -> Map_u:
        """
        Bin the data and the uncertainty by the amount of pixels given.
        Note that this works with every square map, even though the number of pixels to bin cannot fully divide the map's size. In
        the case of a rectangular map, it cannot always find a suitable reshape size.

        Arguments
        ---------
        nb_pix_bin: int. Specifies the number of pixels to be binned together along a single axis. For example, the default value 2
        will give a new map in which every pixel is the mean value of every 4 pixels (2x2 bin).

        Returns
        -------
        Map_u object: binned map.
        """
        binned_data, binned_uncertainties = super().bin_map(nb_pix_bin), super().bin_map(nb_pix_bin, self.uncertainties)
        return Map_u(fits.HDUList([fits.PrimaryHDU(binned_data.data, binned_data.header),
                                   fits.ImageHDU(binned_uncertainties.data, binned_data.header)]))
    
    def smooth_order_change(self, center: int=(527, 484)) -> Map_u:
        """
        Smooth the fitted FWHM of the calibration cube for the first two interference order changes. This is needed as the FWHM
        is reduced at points where the calibration peak changes of interference order. This changes the pixels' value in an
        order change to the mean value of certain pixels in a 7x7 area around the pixel.

        Arguments
        ---------
        center: tuple. Specifies the coordinates of the interference pattern's center pixel.

        Returns
        -------
        Map_u object: map with the smoothed instrumental function.
        """
        data = np.copy(self.data)
        uncertainties = np.copy(self.uncertainties)
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
                    near_pixels = np.copy(self.data[y-3:y+4, x-3:x+4])
                    near_pixels_uncertainty = np.copy(self.uncertainties[y-3:y+4, x-3:x+4])

                    if radiuses[0] - 4*bin_factor <= current_radius <= radiuses[0] + 4*bin_factor:
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[0]] = np.NAN
                    else:
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[1]] = np.NAN
                    
                    data[y,x] = np.nanmean(near_pixels)
                    uncertainties[y,x] = np.nanmean(near_pixels * 0 + near_pixels_uncertainty)
                    # The addition of near_pixels * 0 makes it so the pixels that have np.NAN will not be used
        return Map_u(fits.HDUList([fits.PrimaryHDU(data, self.header),
                                   fits.ImageHDU(uncertainties, self.header)]))

    def reproject_on(self, other: Map_u) -> Map_u:
        """
        Get the reprojection of the map on the other object's WCS. This makes the coordinates match.

        Arguments
        ---------
        other: Map_u object. Reference map to project on and to base the shift of WCS.

        Returns
        -------
        Map_u object: map with WCS aligned to the other map.
        """
        reprojected_data = reproject_interp(self.object[0], other.header, return_footprint=False, order="nearest-neighbor")
        reprojected_uncertainties = reproject_interp(self.object[1], other.header, return_footprint=False, order="nearest-neighbor")
        return Map_u(fits.HDUList([fits.PrimaryHDU(reprojected_data, other.header),
                                   fits.ImageHDU(reprojected_uncertainties, self.header)]))
    
    def align_regions(self) -> Map_u:
        """
        Get the squared FWHM map in which the instrumental function has been subtracted with the three regions corrected to fit
        better the WCS.

        Returns
        -------
        Map_u object: global map with the three aligned regions, result of the subtraction of the squared FWHM map and the squared
        instrumental function map.
        """
        regions = [
            pyregion.open("gaussian_fitting/regions/region_1.reg"),
            pyregion.open("gaussian_fitting/regions/region_2.reg"),
            pyregion.open("gaussian_fitting/regions/region_3.reg")
        ]

        # A mask of zeros and ones is created with the regions
        masks = [region.get_mask(hdu=self.object[0]) for region in regions]
        masks = [np.where(mask == False, 0, 1) for mask in masks]

        # The map's data is removed where a mask applies
        new_map = self.copy() * (1 - (masks[0] + masks[1] + masks[2]))
        # Every specific map needs to have the same values than the global map, but the header is changed to fit a specific region
        # The right headers are first opened, then the values are changed
        specific_headers = [
            Map_u(fits.open(f"gaussian_fitting/maps/reproject/region_1_widening.fits")).header,
            Map_u(fits.open(f"gaussian_fitting/maps/reproject/region_2_widening.fits")).header,
            Map_u(fits.open(f"gaussian_fitting/maps/reproject/region_3_widening.fits")).header
        ]
        # The real data is inserted
        specific_maps = []
        for header in specific_headers:
            specific_maps.append(Map_u(fits.HDUList([fits.PrimaryHDU(np.copy(self.data), header),
                                                     fits.ImageHDU(np.copy(self.uncertainties), header)])))
        # Alignment of the specific maps on the global WCS
        reprojected_specific_maps = [specific_map.reproject_on(self) for specific_map in specific_maps]
        # Only the data within the mask is kept
        region_data = [specific_map * masks[i] for i, specific_map in enumerate(reprojected_specific_maps)]
        new_map += region_data[0] + region_data[1] + region_data[2]
        return new_map

    def transfer_temperature_to_FWHM(self) -> Map_u:
        """
        Get the FWHM of the thermal Doppler broadening. This is used to convert the temperature map into a FWHM map that
        can be compared with other FWHM maps. This method uses the NII peak's wavelength for the Doppler calculations.

        Returns
        -------
        Map_u object: map of the FWHM due to thermal Doppler broadening.
        """
        angstroms_center = 6583.41              # Emission wavelength of NII 
        m = 14.0067 * scipy.constants.u         # Nitrogen mass
        c = scipy.constants.c                   # Light speed
        k = scipy.constants.k                   # Boltzmann constant
        angstroms_FWHM = 2 * float(np.sqrt(2 * np.log(2))) * angstroms_center * (self * k / (c**2 * m))**0.5
        speed_FWHM = c * angstroms_FWHM / angstroms_center / 1000
        return speed_FWHM



class Map_usnr(Map_u):
    """
    Encapsulate the methods specific to maps with uncertainties and signal to noise ratios.
    Note that a Map_usnr is essentially three Map objects into a single object, the first Map being the data, the second one
    being its uncertainty and the third one being the signal to noise ratio. This makes conversion from Map_usnr -> Map easier
    via one of the following statements: data_map, uncertainty_map, snr_map = Map_usnr | snr_map = Map_usnr[2]
    data_map, uncertainty_map and snr_map would then be three Map objects.
    It is also possible to create a Map_usnr object from three Map objects using the from_map_objects method.
    """
    
    def __init__(self, fits_list, name: str=None):
        """
        Initialize a Map_usnr object. 

        Arguments
        ---------
        fits_list: astropy.io.fits.hdu.hdulist.HDUList. List of astropy objects. Contains the values, uncertainties, signal to
        noise ratio and header of the map.
        name: str, default=None. Name of the Map_u. This is primarily used with the Maps object.
        """
        super().__init__(fits_list, name)
        self.snr = fits_list[2].data
        assert self.data.shape == self.snr.shape, "The data and signal to noise ratios sizes do not match."

    @classmethod
    def from_Map_objects(self, map_data: Map, map_uncertainty: Map, map_snr: Map) -> Map_usnr:
        """
        Create a Map_usnr object using three Map objects. An object may be created using the following
        statement: new_map = Map_usnr.from_Map_objects(data_map, uncertainties_map, snr_map),
        where data_map, uncertainties_map and snr_map are three Map objects.

        Arguments
        ---------
        map_data: Map object. Serves as the Map_usnr's data.
        map_uncertainty: Map object. Serves as the Map_usnr's uncertainties.
        map_snr: Map object. Serves as the Map_usnr's signal to noise ratio.

        Returns
        -------
        Map_usnr object: map with the corresponding data and uncertainties.
        """
        return self(fits.HDUList([fits.PrimaryHDU(map_data.data, map_data.header),
                                  fits.ImageHDU(map_uncertainty.data, map_data.header),
                                  fits.ImageHDU(map_snr.data, map_data.header)]))
    
    @classmethod
    def from_Map_u_object(self, map_values: Map_u, map_snr: Map) -> Map_usnr:
        """
        Create a Map_usnr object using a Map_u object and a Map object. An object may be created
        using the following statement: new_map = Map_usnr.from_Map_u_object(map_values, snr_map),
        where map_values is a Map_u and snr_map is a Map object.

        Arguments
        ---------
        map_values: Map_u object. Serves as the Map_usnr's data, uncertainties and header.
        map_snr: Map object. Serves as the Map_usnr's signal to noise ratio.

        Returns
        -------
        Map_usnr object: map with the corresponding data and uncertainties.
        """
        return self(fits.HDUList([fits.PrimaryHDU(map_values.data, map_values.header),
                                  fits.ImageHDU(map_values.uncertainties, map_values.header),
                                  fits.ImageHDU(map_snr.data, map_values.header)]))
    
    def __eq__(self, other):
        return super().__eq__(other) and np.nansum(self.snr - other.snr) == 0.
    
    def copy(self):
        return self.from_Map_u_object(super().copy(), self.snr)
    
    def save_as_fits_file(self, filename: str):
        """
        Write the Map_usnr as a fits file of the specified name with or without a header. If the object has a header, it will be
        saved. The data uncertainty is saved as the [1] extension of the fits file and the signal to noise ratio is saved as the
        [2] extension of the. To view the uncertainty and signal to noise ratio maps on DS9, simply open the file with the
        following path: File -> Open as -> Multi Extension Cube. The data, its uncertainty and its snr will then be visible just
        like in a data cube.
        
        Arguments
        ---------
        filename: str. Indicates the path and name of the created file. If the file already exists, a warning will appear and the
        file can be overwritten.
        """
        # Check if the file already exists
        try:
            fits.open(filename)[0]
            # The file already exists
            while True:
                answer = input(f"The file '{filename}' already exists, do you wish to overwrite it ? [y/n]")
                if answer == "y":
                    hdu_list = fits.HDUList([
                        fits.PrimaryHDU(self.data, self.header),
                        fits.ImageHDU(self.uncertainties, self.header),
                        fits.ImageHDU(self.snr, self.header)
                    ])

                    hdu_list.writeto(filename, overwrite=True)
                    print("File overwritten.")
                    break

                elif answer == "n":
                    break
                
        except:
            # The file does not yet exist
            hdu_list = fits.HDUList([
                fits.PrimaryHDU(self.data, self.header),
                fits.ImageHDU(self.uncertainties, self.header),
                fits.ImageHDU(self.snr, self.header)
            ])

            hdu_list.writeto(filename, overwrite=True)

    def filter_snr(self, snr_threshold: float=6) -> Map_usnr:
        """ 
        Filter the map's values by keeping only the data that has a signal to noise ratio superior or equal to the provided
        quantity.

        Arguments
        ---------
        snr_threshold: float, default=6. The pixels that have a signal to noise ratio inferior to this quantity will be
        excluded.

        Returns
        Map_usnr object: map with the filtered data.
        """
        mask = np.ma.masked_less(self.snr, snr_threshold).mask
        mask = np.where(mask == True, 0, 1)
        return Map_usnr(fits.HDUList([fits.PrimaryHDU(self.data * mask, self.header),
                                      fits.ImageHDU(self.uncertainties * mask, self.header),
                                      fits.ImageHDU(self.snr * mask, self.header)]))



class Maps():
    """ 
    Encapsulates the methods that are specific to a multitude of linked maps. This class is mainly used as the output of the
    fit() method and allows for many convenient operations.
    """

    def __init__(self, maps: list):
        self.content = {}
        self.names = {}
        for i, individual_map in enumerate(maps):
            self.content[individual_map.name] = individual_map
            self.names[i] = individual_map.name
    
    @classmethod
    def open_from_folder(self, folder_path) -> Maps:
        maps = []
        files_in_folder = os.listdir(folder_path)
        for file in files_in_folder:
            if file[-5:] == ".fits":
                name = file[:-5]
                try:
                    maps.append(Map_usnr(fits.open(f"{folder_path}/{file}"), name=name))
                except:
                    try:
                        maps.append(Map_u(fits.open(f"{folder_path}/{file}"), name=name))
                    except:
                        maps.append(Map(fits.open(f"{folder_path}/{file}")[0], name=name))
        return self(maps)

    def __iter__(self):
        self.n = -1
        return self
    
    def __next__(self):
        self.n += 1
        if self.n > len(self.content) - 1:
            raise StopIteration
        else:
            return self.content[self.names[self.n]]

    def __getitem__(self, map_name):
        return self.content[map_name]
    
    def save_as_fits_file(self, folder_path: str):
        for name, element in self.content.items():
            element.save_as_fits_file(f"{folder_path}/{name}.fits")

