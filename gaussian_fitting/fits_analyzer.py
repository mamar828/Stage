import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings

from astropy.io import fits
from astropy.wcs import WCS

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
        header_copy["CDELT1"] *= nb_pix_bin
        header_copy["CDELT2"] *= nb_pix_bin
        header_copy["CRPIX1"] /= nb_pix_bin
        header_copy["CRPIX2"] /= nb_pix_bin
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
                answer = input(f"The file '{filename}' already exists, do you wish to overwrite it ? [yes/no]")
                if answer == "yes":
                    fits.writeto(filename, self.data, self.header, overwrite=True)
                    print("File overwritten.")
                    break
                elif answer == "no":
                    break        
                
        except:
            # The file does not yet exist
            fits.writeto(filename, self.data, self.header, overwrite=True)



class Data_cube(Fits_file):
    """
    Encapsulate all the useful methods for the analysis of a data cube.
    """

    def __init__(self, data_cube_file_name=None):
        """
        Initialize an analyzer object. The datacube's file name must be given.

        Arguments
        ---------
        data_cube_file_name: str, optional. Specifies the path of the file inside the current folder.
        """
        if data_cube_file_name is not None:
            self.data = fits.open(data_cube_file_name)[0].data
            self.header = fits.open(data_cube_file_name)[0].header

    def fit_calibration(self, data_cube=None):
        """
        Fit the whole data cube as if it was a calibration cube and extract the FWHM and its uncertainty at every point.
        Set the global numpy array self.fit_fwhm_map the value of the fit's FWHM at every point. The first element of the
        array along the third axis is the FWHM value and the second value is its uncertainty. Print the x value of the
        pixel row whose x is divisible by 10 and print a point for every other row. Every print is a row being fitted.
        Note that this process utilizes only a single CPU and therefore could be accelerated.

        Arguments
        ---------
        data: numpy array, optional. Specifies the data cube to be fitted. If None, the data cube of the analyzer will be
        fitted directly.

        Returns
        -------
        numpy array: map of the fit's FWHM and its uncertainty at every point.
        """
        if data_cube is None:
            data_cube = np.copy(self.data)
        
        self.fit_fwhm_map = np.zeros([data_cube.shape[1], data_cube.shape[2], 2])
        for x in range(0, data_cube.shape[2]):
            # Optional prints
            if x%10 == 0:
                print("\n", x, end=" ")
            else:
                print(".", end="")
            for y in range(data_cube.shape[1]):
                try:
                    spectrum_object = Spectrum(data_cube[:,y,x], calibration=True)
                    spectrum_object.fit()
                    self.fit_fwhm_map[y,x,:] = spectrum_object.get_FWHM_speed(
                        spectrum_object.get_fitted_gaussian_parameters(), spectrum_object.get_uncertainties()["g0"]["stddev"])
                except:
                    self.fit_fwhm_map[y,x,:] = [np.NAN, np.NAN]
        
        # In the numpy array, every vertical group is a y coordinate, starting from (1,1) at the top
        # Every element in a group is a x coordinate
        # Every sub-element is the fwhm and its uncertainty
        return self.fit_fwhm_map

    def fit_NII(self):
        """
        Fit the whole data cube to extract the NII gaussian's FWHM. This method presupposes that four OH peaks and one Halpha
        peaks are present in the cube's spectrum in addition to the NII peak.
        WARNING: Due to the use of the multiprocessing library, calls to this function NEED to be made inside a condition
        state with the following phrasing:
        if __name__ == "__main__":
        This prevents the code to create recursively instances of this code that would eventually overload the CPUs.

        Returns
        -------
        numpy array: map of the NII peak's FWHM at every point. Each coordinate is another list with the first element being
        the FWHM value and the second being its uncertainty.
        """
        data = self.bin_cube(self.data, 2)
        fit_fwhm_list = []
        # This automatically generates an optimal number of workers
        pool = multiprocessing.Pool()
        start = time.time()
        fit_fwhm_list.append(np.array(pool.map(worker_fit, list((y, data) for y in range(data.shape[1])))))
        stop = time.time()
        print(stop-start, "s")
        pool.close()
        return np.squeeze(np.array(fit_fwhm_list), axis=0)
        
    def bin_cube(self, nb_pix_bin=2):
        """
        Bin a specific cube by the amount of pixels given for every channel.
        Note that this works with every square cube, even though the number of pixels to bin cannot fully divide the cube's size. In
        the case of a rectangular cube, it cannot always find a suitable reshape size.

        Arguments
        ---------
        nb_pix_bin: int. Specifies the number of pixels to be binned together along a single axis. For example, the default value 2
        will give a new cube in which every pixel at a specific channel is the mean value of every 4 pixels (2x2 bin) at that same
        channel.

        Returns
        -------
        numpy array: binned cube.
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
        return np.nanmean(bin_array, axis=(2,4))

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

    # ---------------------------------- MAY BECOME OBSOLETE ----------------------------------
    def get_corrected_width(self, fwhm_NII=np.ndarray, fwhm_NII_uncertainty=np.ndarray,
                            instrumental_function_width=np.ndarray, instrumental_function_width_uncertainty=np.ndarray):
        """
        Get the fitted gaussian FWHM value corrected by removing the instrumental_function_width.

        Arguments
        ---------
        fwhm_NII: numpy array. Map of the FWHM of the NII gaussian.
        fwhm_NII_uncertainty: numpy array. Uncertainty map of the FWHM of the NII gaussian.
        instrumental_function_width: numpy array. Map of the FWHM of the calibration peak.
        instrumental_function_width_uncertainty: numpy array. Uncertainty map of the FWHM of the calibration peak.

        Returns
        -------
        numpy array: map of the calibration FWHM subtracted to the NII fwhm. The first element at a specific coordinate is
        the corrected FWHM value and the second element is its uncertainty.
        """
        return [fwhm_NII - instrumental_function_width,
                fwhm_NII_uncertainty + instrumental_function_width_uncertainty]
    

def worker_fit(args):
    """
    Fit an entire line of the NII cube.

    Arguments
    ---------
    args: tuple. The first element is the y value of the line to be fitted and the second element is the data_cube used.
    Note that arguments are given in tuple due to the way the multiprocessing library operates.

    Returns
    -------
    list: FWHM value of the fitted gaussian on the NII peak along the specified line. Each coordinates has two value: the former
    being the FWHM value whilst the latter being its uncertainty.
    """
    y, data = args
    line = []
    for x in range(data.shape[1]):
        spectrum_object = Spectrum(data[:,y,x], calibration=False)
        spectrum_object.fit(spectrum_object.get_initial_guesses())
        line.append(spectrum_object.get_FWHM_speed(
                    spectrum_object.get_fitted_gaussian_parameters()[4], spectrum_object.get_uncertainties()["g4"]["stddev"]))
    return line



class Map(Fits_file):
    """
    Encapsulate the necessary methods to compare and treat maps.
    """

    def __init__(self, map_file_name=None):
        """
        Initialize a map object.

        Arguments
        ---------
        map_data: str, optional. Path of the map's data.
        """
        if map_file_name is not None:
            self.data = fits.open(map_file_name)[0].data
            self.header = fits.open(map_file_name)[0].header

    def __add__(self, other):
        assert self.data.shape == other.data.shape, "Different map sizes are being added."
        return self.data + other.data
    
    def __sub__(self, other):
        assert self.data.shape == other.data.shape, "Different map sizes are being added."
        return self.data - other.data

    def __pow__(self):
        return self.data ** 2
    
    def sqrt(self):
        return np.sqrt(self.data)

    def plot_map(self, color_autoscale=True, bounds=None):
        """
        Plot the map in matplotlib.pyplot.

        Arguments
        ---------
        color_autoscale: bool. If True, the colorbar will automatically scale to have as bounds the map's minimum and maximum. If
        False, bounds must be specified.
        bounds: tuple. Indicates the colorbar's bounds. The tuple's first element is the minimum and the second is the maximum.
        """
        data = np.copy(self.data)
        if color_autoscale:
            plt.colorbar(plt.imshow(data, origin="lower", cmap="viridis"))
        elif bounds:
            plt.colorbar(plt.imshow(data, origin="lower", cmap="viridis", vmin=bounds[0], vmax=bounds[1]))
        else:
            plt.colorbar(plt.imshow(data, origin="lower", cmap="viridis", vmin=data[round(data.shape[0]/2), round(data.shape[1]/2)]*3/5,
                                                                     vmax=data[round(data.shape[0]/10), round(data.shape[1]/10)]*2))
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
        numpy array: binned map.
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
        return np.nanmean(bin_array, axis=(1,3))
    
    def smooth_order_change(self, uncertainty_map, center=(527, 484)):
        """
        Smooth the fitted FWHM of the calibration cube for the first two interference order changes. This is needed as the FWHM is
        reduced at points where the calibration peak changes of interference order. This changes the pixels' value in an order
        change to the mean value of certain pixels in a 7x7 area around the pixel.

        Arguments
        ---------
        uncertainty_map: Map object. Map of the FWHM's uncertainty of the fitted peak at every point.
        center: tuple. Specifies the coordinates of the interference pattern's center pixel.

        Returns
        -------
        numpy array: map of the FWHM at every point and its associated uncertainty.
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
        return np.stack((data, uncertainties), axis=2)

    def reproject(self, other):
        pass

    def get_thermal_FWHM(self):
        
    
