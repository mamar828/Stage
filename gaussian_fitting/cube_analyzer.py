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


class Data_cube_analyzer():
    """
    Encapsulate all the useful methods for the analysis of a data cube.
    """

    def __init__(self, data_cube_file_name=str):
        """
        Initialize an analyzer object. The datacube's file name must be given.

        Arguments
        ---------
        data_cube_file_name: str. Specifies the path of the file inside the current folder.
        """
        self.data_cube = fits.open(data_cube_file_name)[0].data

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
            data_cube = self.data_cube
        
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

    def save_as_fits_file(self, filename=str, array=np.ndarray, header=None):
        """
        Write an array as a fits file of the specified name with or without a header.

        Arguments
        ---------
        filename: str. Indicates the path and name of the created file. If the file already exists, it is overwritten.
        array: numpy array. Feeds the array to be saved as a fits file.
        header: astropy.io.fits.header.Header, optional. If specified, the fits file will have the given header. This is mainly
        useful for saving maps with usable WCS.
        """
        fits.writeto(filename, array, header, overwrite=True)
    
    def plot_map(self, map=np.ndarray, color_autoscale=True, bounds=None):
        """
        Plot a given map in matplotlib.pyplot.

        Arguments
        ---------
        map: numpy array. Gives the array that needs to be plotted.
        color_autoscale: bool. If True, the colorbar will automatically scale to have as bounds the map's minimum and maximum. If
        False, bounds must be specified.
        bounds: tuple. Indicates the colorbar's bounds. The tuple's first element is the minimum and the second is the maximum.
        """
        if color_autoscale:
            plt.colorbar(plt.imshow(map, origin="lower", cmap="viridis"))
        elif bounds:
            plt.colorbar(plt.imshow(map, origin="lower", cmap="viridis", vmin=bounds[0], vmax=bounds[1]))
        else:
            plt.colorbar(plt.imshow(map, origin="lower", cmap="viridis", vmin=map[round(map.shape[0]/2), round(map.shape[1]/2)]*3/5,
                                                                     vmax=map[round(map.shape[0]/10), round(map.shape[1]/10)]*2))
        plt.show()

    def bin_map(self, map=np.ndarray, nb_pix_bin=2):
        """
        Bin a specific map by the amount of pixels given.
        Note that this works with every square map, even though the number of pixels to bin cannot fully divide the map's size. In
        the case of a rectangular map, it cannot always find a suitable reshape size.

        Arguments
        ---------
        map: numpy array. Gives the array that needs to be binned.
        nb_pix_bin: int. Specifies the number of pixels to be binned together along a single axis. For example, the default value 2
        will give a new map in which every pixel is the mean value of every 4 pixels (2x2 bin).

        Returns
        -------
        numpy array: binned map.
        """
        # Loop over the nb_pix_bin to find the number of pixels that needs to be cropped
        for i in range(nb_pix_bin):
            try:
                # Create a 4 dimensional array that regroups every group of pixels (2 times the nb_pix_bin) into a new grid whose
                # size has been divided by the number of pixels to bin
                bin_array = map.reshape(int(map.shape[0]/nb_pix_bin), nb_pix_bin, int(map.shape[1]/nb_pix_bin), nb_pix_bin)
                break
            except ValueError:
                # This error occurs if the nb_pix_bin integer cannot fully divide the map's size
                print(f"Map to bin will be cut by {i+1} pixel(s).")
                map = map[:-1,:-1]
        # The mean value of every pixel group is calculated and the array returns to a two dimensional state
        return np.nanmean(bin_array, axis=(1,3))
    
    def bin_cube(self, cube=np.ndarray, nb_pix_bin=2):
        """
        Bin a specific cube by the amount of pixels given for every channel.
        Note that this works with every square cube, even though the number of pixels to bin cannot fully divide the cube's size. In
        the case of a rectangular cube, it cannot always find a suitable reshape size.

        Arguments
        ---------
        map: numpy array. Gives the array that needs to be binned.
        nb_pix_bin: int. Specifies the number of pixels to be binned together along a single axis. For example, the default value 2
        will give a new cube in which every pixel at a specific channel is the mean value of every 4 pixels (2x2 bin) at that same
        channel.

        Returns
        -------
        numpy array: binned cube.
        """
        # Loop over the nb_pix_bin to find the number of pixels that needs to be cropped
        for i in range(nb_pix_bin):
            try:
                # Create a 5 dimensional array that regroups, for every channel, every group of pixels (2 times the nb_pix_bin)
                # into a new grid whose size has been divided by the number of pixels to bin
                bin_array = cube.reshape(cube.shape[0], int(cube.shape[1]/nb_pix_bin), nb_pix_bin,
                                                        int(cube.shape[2]/nb_pix_bin), nb_pix_bin)
                break
            except ValueError:
                # This error occurs if the nb_pix_bin integer cannot fully divide the cube's size
                print(f"Cube to bin will be cut by {i+1} pixel(s).")
                cube = cube[:,:-1,:-1]
        # The mean value of every pixel group at every channel is calculated and the array returns to a three dimensional state
        return np.nanmean(bin_array, axis=(2,4))

    def smooth_order_change(self, data_array=np.ndarray, uncertainty_array=np.ndarray, center=tuple):
        """
        Smooth the fitted FWHM of the calibration cube for the first two interference order changes. This is needed as the FWHM is
        reduced at points where the calibration peak changes of interference order.

        Arguments
        ---------
        data_array: numpy array. Gives the FWHM of the fitted peak at every point.
        uncertainty_array: numpy array. Gives the FWHM's uncertainty of the fitted peak at every point.
        center: tuple. Specifies the coordinates of the interference pattern's center pixel.

        Returns
        -------
        numpy array: map of the FWHM at every point and its associated uncertainty.
        """
        center = round(center[0]), round(center[1])
        # The bin_factor corrects the distances in the case of a binned array
        bin_factor = center[0] / 527
        # The smoothing_max_thresholds is defined by trial and error is 
        smoothing_max_thresholds = [0.4, 1.8]
        bounds = [
            np.array((255,355)) * bin_factor,
            np.array((70,170)) * bin_factor
        ]
        peak_regions = [
            list(data_array[center[1], int(bounds[0][0]):int(bounds[0][1])]),
            list(data_array[center[1], int(bounds[1][0]):int(bounds[1][1])])
        ]
        radiuses = [
            center[0] - (peak_regions[0].index(min(peak_regions[0])) + 255),
            center[0] - (peak_regions[1].index(min(peak_regions[1])) + 70)
        ]

        smooth_data = np.copy(data_array)
        smooth_uncertainties = np.copy(uncertainty_array)

        for x in range(data_array.shape[1]):
            for y in range(data_array.shape[0]):
                current_radius = np.sqrt((x-center[0])**2 + (y-center[1])**2)

                if (radiuses[0] - 5*bin_factor <= current_radius <= radiuses[0] + 5*bin_factor or
                    radiuses[1] - 4*bin_factor <= current_radius <= radiuses[1] + 4*bin_factor):
                    near_pixels = np.copy(data_array[y-3:y+4, x-3:x+4])
                    near_pixels_uncertainty = np.copy(uncertainty_array[y-3:y+4, x-3:x+4])

                    if radiuses[0] - 4*bin_factor <= current_radius <= radiuses[0] + 4*bin_factor:
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[0]] = np.NAN
                    else:
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[1]] = np.NAN
                    
                    smooth_data[y,x] = np.nanmean(near_pixels)
                    smooth_uncertainties[y,x] = np.nanmean(near_pixels * 0 + near_pixels_uncertainty)
        return np.stack((smooth_data, smooth_uncertainties), axis=2)

    def get_center_point(self):
        center_guess = 527, 484
        distances = {"x": [], "y": []}
        success = 0
        for channel in range(1,49):
            channel_dist = {}
            intensity_max = {
                "intensity_max_x": self.data_cube[channel-1, center_guess[1], :],
                "intensity_max_y": self.data_cube[channel-1, :, center_guess[0]]
            }
            for name, axe_list in intensity_max.items():
                axes_pos = []
                for coord in range(1, 1023):
                    if (axe_list[coord-1] < axe_list[coord] > axe_list[coord+1]
                         and axe_list[coord] > 500):
                        axes_pos.append(coord)
                # Verification of single maxes
                for i in range(len(axes_pos)-1):
                    if axes_pos[i+1] - axes_pos[i] < 50:
                        if axe_list[axes_pos[i]] > axe_list[axes_pos[i+1]]:
                            axes_pos[i+1] = 0
                        else:
                            axes_pos[i] = 0
                axes_pos = np.setdiff1d(axes_pos, 0)
                
                if len(axes_pos) == 4:
                    channel_dist[name] = [(axes_pos[3] - axes_pos[0])/2 + axes_pos[0], (axes_pos[2] - axes_pos[1])/2 + axes_pos[1]]
                    distances[name[-1]].append(channel_dist[name])
                    success += 1
        x_mean, y_mean = np.mean(distances["x"], axis=(0,1)), np.mean(distances["y"], axis=(0,1))
        x_uncertainty, y_uncertainty = np.std(distances["x"], axis=(0,1)), np.std(distances["y"], axis=(0,1))
        return [x_mean, x_uncertainty], [y_mean, y_uncertainty]

    def get_corrected_width(self, fwhm_NII, fwhm_NII_uncertainty,
                            instrumental_function_width, instrumental_function_width_uncertainty):
        return [fwhm_NII - instrumental_function_width,
                fwhm_NII_uncertainty + instrumental_function_width_uncertainty]
    

def worker_fit(args):
    y, data = args
    line = []
    for x in range(data.shape[1]):
        spectrum_object = Spectrum(data[:,y,x], calibration=False)
        spectrum_object.fit(spectrum_object.get_initial_guesses())
        line.append(spectrum_object.get_FWHM_speed(
                    spectrum_object.get_fitted_gaussian_parameters()[4], spectrum_object.get_uncertainties()["g4"]["stddev"]))
    return line

"""
if __name__ == "__main__":
    analyzer = Data_cube_analyzer("night_34.fits")
    data = analyzer.bin_cube(analyzer.data_cube, 2)
    fit_fwhm_list = []
    pool = multiprocessing.Pool()
    start = time.time()
    fit_fwhm_list.append(np.array(pool.map(worker_fit, list((y, data) for y in range(data.shape[1])))))
    stop = time.time()
    print(stop-start, "s")
    pool.close()
    fitted_array = np.squeeze(np.array(fit_fwhm_list), axis=0)
    # analyzer.save_as_fits_file("maps/fwhm_NII.fits", fitted_array[:,:,0])
    # analyzer.save_as_fits_file("maps/fwhm_NII_unc.fits", fitted_array[:,:,1])
"""



# file = fits.open("cube_NII_Sh158_with_header.fits")[0].data
fwhms = fits.open("maps/fwhm_NII.fits")[0].data
fwhms_unc = fits.open("maps/fwhm_NII_unc.fits")[0].data
calibs = fits.open("maps/smoothed_instr_f.fits")[0].data
calibs_unc = fits.open("maps/smoothed_instr_f_unc.fits")[0].data
corrected_fwhm = fits.open("maps/corrected_fwhm.fits")[0].data
corrected_fwhm_unc = fits.open("maps/corrected_fwhm_unc.fits")[0].data

# header = fits.open("night_34.fits")[0].header
# print(header)

# analyzer = Data_cube_analyzer("night_34.fits")
# analyzer.plot_map(corrected_fwhm, color_autoscale=False, bounds=(0,50))

# sp = Spectrum(analyzer.bin_cube(analyzer.data_cube)[:,223,309], calibration=False)
# sp.fit(sp.get_initial_guesses())
# sp.plot_fit()

# corrected_map = analyzer.get_corrected_width(fwhms, fwhms_unc, analyzer.bin_map(calibs), analyzer.bin_map(calibs_unc))
# analyzer.save_as_fits_file("maps/corrected_fwhm.fits", corrected_map[0])
# analyzer.save_as_fits_file("maps/corrected_fwhm_unc.fits", corrected_map[1])


hawc = fits.open("night_34.fits")
a = Data_cube_analyzer("night_34.fits")

header_0 = (hawc[0].header).copy()
header_0["CDELT1"] = header_0["CDELT1"] * 2
header_0["CDELT2"] = header_0["CDELT2"] * 2
header_0["CRPIX1"] = header_0["CRPIX1"] / 2
header_0["CRPIX2"] = header_0["CRPIX2"] / 2
# header_0.update(NAXIS1=512, NAXIS2=512)
print(type(header_0))
# a.save_as_fits_file("maps/ba_corrected_fwhm.fits", corrected_fwhm, header_0)
