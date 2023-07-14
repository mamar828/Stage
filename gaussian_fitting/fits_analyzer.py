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
        Give the user an update of the status of the running code in the text file output.txt and in a print.

        Arguments
        ---------
        info: str. Beggining string to give information about the current running program.
        """
        try:
            # The file is opened and the last number is extracted
            file = open("output.txt", "r")
            number = int(file.readlines()[-1])
            file = open("output.txt", "w")
            if number == 0:
                # First print is aligned so that all dots are justified
                print(" " * len(info.split("/")[-1]), end="", flush=True)
            new_num = number + 1
            file.write(f"{info}\n{str(new_num)}")
            file.close()
            if new_num%10 == 0:
                # At every 10 rows that is fitted, a newline is printed with the number of the current line
                # The number also needs to be aligned for the dots to be justified
                alignment_string = " " * (len(info.split("/")[-1]) - len(str(new_num)))
                print(f"\n{alignment_string}{new_num}", end="", flush=True)
            else:
                print(".", end="", flush=True)
        except:
            # Sometimes the read method is unsuccessful
            pass



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
        try:
            self.object = fits_object
            self.data = fits_object.data
            self.header = fits_object.header
        except:
            self.object = fits_object[0]
            self.data = fits_object[0].data
            self.header = fits_object[0].header
        
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
        print(f"Number of processes used: {pool._processes}")
        self.reset_update_file()
        start = time.time()
        fit_fwhm_list.append(np.array(pool.map(worker_fit, list((y, data, "calibration", self.header) 
                                                                 for y in range(data.shape[1])))))
        stop = time.time()
        print("\nFinished in", stop-start, "s.")
        pool.close()
        new_header = self.get_header_without_third_dimension()
        # The map is temporarily stored in a simple format to facilitate extraction
        fit_fwhm_map = np.squeeze(np.array(fit_fwhm_list), axis=0)
        return Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0], new_header),
                                   fits.ImageHDU(fit_fwhm_map[:,:,1], new_header)]))

    def fit_all(self, extract: list, seven_components_fit_authorized: bool=False) -> tuple:
        """
        Fit the whole data cube to extract the peaks' data. This method presupposes that four OH peaks, one Halpha peak and
        one NII peak (sometimes two) are present.
        WARNING: Due to the use of the multiprocessing library, calls to this function NEED to be made inside a condition
        state with the following phrasing:
        if __name__ == "__main__":
        This prevents the code to recursively create instances of itself that would eventually overload the CPUs.

        Arguments
        ---------
        extract: list. Names of the gaussians' parameters to extract. Supported terms are: "mean", "amplitude" and "FWHM".
        Any combination or number of these terms can be given.
        seven_components_fit_authorized: bool, default=False. Specifies if double NII peaks are considered. If True, the fitting
        may detect two components and fit these separately. The NII values in the returned maps for a certain parameter will then
        be the mean value of the two fits.

        Returns
        -------
        tuple of Maps object: the Maps object representing every ray's mean, amplitude or FWHM are returned in the order they
        were put in argument, thus the tuple may have a length of 1, 2 or 3. Every Maps object has the maps of every ray present
        in the provided data cube. 
        Note that each map is a Map_usnr object when computing the FWHM whereas the maps are Map_u objects when computing
        amplitude or mean. In any case, in every Maps object is a Map object having the value 1 when a seven components fit was
        executed and 0 otherwise.
        """
        # The provided extract data is verified before fitting
        assert extract != [], "At least a parameter to be extracted should be provided in the extract list."
        assert isinstance(extract, list), f"Extract argument must be a list, not {type(extract).__name__}."
        for element in extract:
            assert element == "mean" or element == "amplitude" or element == "FWHM", "Unsupported element in extract list."

        data = np.copy(self.data)
        fit_fwhm_list = []
        pool = multiprocessing.Pool()           # This automatically generates an optimal number of workers
        print(f"Number of processes used: {pool._processes}")
        self.reset_update_file()
        start = time.time()
        if seven_components_fit_authorized:
            fit_fwhm_list.append(np.array(pool.map(worker_fit, list((y, data, "NII_2", self.header)
                                                                    for y in range(data.shape[1])))))
        else:
            fit_fwhm_list.append(np.array(pool.map(worker_fit, list((y, data, "NII_1", self.header)
                                                                    for y in range(data.shape[1])))))
        stop = time.time()
        print("\nFinished in", stop-start, "s.")
        pool.close()
        new_header = self.get_header_without_third_dimension()
        # The list containing the fit results is transformed into a numpy array to facilitate extraction
        fit_fwhm_array = np.squeeze(fit_fwhm_list)
        # The fit_fwhm_array has 5 dimensions (x_shape,y_shape,3,7,3) and the last three dimensions are given at every pixel
        # Third dimension: all three gaussian parameters. 0: fwhm, 1: amplitude, 2: mean
        # Fourth dimension: all rays in the data_cube. 0: OH1, 1: OH2, 2: OH3, 3: OH4, 4: NII, 5: Ha, 6: 7 components fit map
        # Fifth dimension: 0: data, 1: uncertainties, 2: snr (only when associated to fwhm values)
        # The 7 components fit map is a map taking the value 0 if a single component was fitted onto NII and the value 1 if
        # two components were considered
        fwhm_maps = Maps([
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,0,0,0], new_header),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,0,1]),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,0,2])]), name="OH1_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,0,1,0], new_header),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,1,1]),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,1,2])]), name="OH2_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,0,2,0], new_header),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,2,1]),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,2,2])]), name="OH3_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,0,3,0], new_header),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,3,1]),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,3,2])]), name="OH4_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,0,4,0], new_header),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,4,1]),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,4,2])]), name="NII_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,0,5,0], new_header),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,5,1]),
                                   fits.ImageHDU(fit_fwhm_array[:,:,0,5,2])]), name="Ha_fwhm"),
            Map(fits.PrimaryHDU(fit_fwhm_array[:,:,0,6,0], new_header), name="7_components_fit")
        ])
        amplitude_maps = Maps([
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,1,0,0], new_header),
                                fits.ImageHDU(fit_fwhm_array[:,:,1,0,1])]), name="OH1_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,1,1,0], new_header),
                                fits.ImageHDU(fit_fwhm_array[:,:,1,1,1])]), name="OH2_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,1,2,0], new_header),
                                fits.ImageHDU(fit_fwhm_array[:,:,1,2,1])]), name="OH3_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,1,3,0], new_header),
                                fits.ImageHDU(fit_fwhm_array[:,:,1,3,1])]), name="OH4_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,1,4,0], new_header),
                                fits.ImageHDU(fit_fwhm_array[:,:,1,4,1])]), name="NII_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,1,5,0], new_header),
                                fits.ImageHDU(fit_fwhm_array[:,:,1,5,1])]), name="Ha_amplitude"),
            Map(fits.PrimaryHDU(fit_fwhm_array[:,:,1,6,0], new_header), name="7_components_fit")
        ])
        mean_maps = Maps([
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,2,0,0], new_header),
                                fits.ImageHDU(fit_fwhm_array[:,:,2,0,1])]), name="OH1_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,2,1,0], new_header),
                                fits.ImageHDU(fit_fwhm_array[:,:,2,1,1])]), name="OH2_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,2,2,0], new_header),
                                fits.ImageHDU(fit_fwhm_array[:,:,2,2,1])]), name="OH3_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,2,3,0], new_header),
                                fits.ImageHDU(fit_fwhm_array[:,:,2,3,1])]), name="OH4_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,2,4,0], new_header),
                                fits.ImageHDU(fit_fwhm_array[:,:,2,4,1])]), name="NII_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_array[:,:,2,5,0], new_header),
                                fits.ImageHDU(fit_fwhm_array[:,:,2,5,1])]), name="Ha_mean"),
            Map(fits.PrimaryHDU(fit_fwhm_array[:,:,2,6,0], new_header), name="7_components_fit")
        ])
        parameter_names = {"FWHM": fwhm_maps, "amplitude": amplitude_maps, "mean": mean_maps}
        return_list = []
        for element in extract:
            return_list.append(parameter_names[element])
        if len(extract) == 1:
            # If only a single Maps is present, the element itself needs to be returned and not a tuple
            return return_list[-1]
        return tuple(return_list)
    
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
    
    def get_flux_map(self) -> Map:
        """
        Get the flux map which corresponds to the sum on every pixel of the multiplication of the intensity by the channel spacing
        at every channel. This is equation 2 which gives the moment-0
        
        Returns
        -------
        Map object: map of the flux at every pixel.
        """
        speed_per_channel = self.header["CDELT3"]
        flux_at_every_channel = self.data * speed_per_channel
        total_flux = np.sum(flux_at_every_channel, axis=0)
        return Map(fits.PrimaryHDU(total_flux, self.get_header_without_third_dimension()))
    
    def get_flux_weighted_centroid_velocity(self) -> Map:
        """
        Get the intensity-weighted centroid velocity of a data cube by the sum on every pixel of the multiplication of the channel
        spacing, speed and intensity at every channel. This is equation 3 which gives the moment-1.
        
        Returns
        -------
        Map object: map of the weighted velocity.
        """
        speed_channel_1 = self.header["CRVAL3"]
        speed_per_channel = self.header["CDELT3"]
        # A list is first created which links every channel to their velocity in km/s
        speed_at_every_channel = np.array([speed_channel_1 + i*speed_per_channel for i in range(48)])
        # The m_s variable corresponds the data cube's dimensions without the spectral axis
        m_s = self.data.shape[1:]
        # The multiplication array is a 3D array with the speed corresponding to every channel
        multiplication_array = np.tile(speed_at_every_channel, m_s[0]*m_s[1]).reshape(m_s[1],m_s[0],48).swapaxes(0,2)
        numerator_data = np.sum(self.data * multiplication_array * speed_per_channel, axis=0)
        numerator = Map(fits.PrimaryHDU(numerator_data, self.get_header_without_third_dimension()))
        denominator = self.get_flux_map()
        return numerator / denominator



def worker_fit(args: tuple) -> list:
    """
    Fit an entire line of a Data_cube.

    Arguments
    ---------
    args: tuple. The first element is the y value of the line to be fitted, the second element is the data of the Data_cube used,
    the third element is a string specifying if the cube is a calibration cube or a NII cube and if a double NII components fit
    should be made: "calibration", "NII_1" or "NII_2" and the fourth element is the Data_cube's header.
    Note that arguments are given in tuple due to the way the multiprocessing library operates.

    Returns
    -------
    list: FWHM value or amplitude of the fitted gaussians at every point along the specified line. In the case of the calibration
    cube, each element is a list of the FWHM and its uncertainty. In the case of the NII cube, each coordinates has seven values:
    the first six are the peaks' FWHM with their uncertainty and signal to noise ratio and the last one is a map indicating where
    fits with sevent components were done. The last map outputs 0 for a six components fit and 1 for a seven components fit.
    """
    y, data, cube_type, header = args
    line = []
    if cube_type == "calibration":
        # A single fit will be made and the FWHM value will be extracted
        for x in range(data.shape[2]):
            spec = Spectrum(data[:,y,x], header, calibration=True)
            spec.fit_calibration()
            try:
                line.append(spec.get_FWHM_speed())
            except:
                line.append([np.NAN, np.NAN])
        Data_cube.give_update(None, f"Calibration fitting progress /{data.shape[1]}")

    elif cube_type[:3] == "NII":
        if cube_type[-1] == "1":
            # A multi-gaussian fit with 6 components will be made and every values will be extracted
            seven_components = False
        elif cube_type[-1] == "2":
            # A multi-gaussian fit with 7 components will be made and every values will be extracted
            seven_components = True
        for x in range(data.shape[2]):
            spec = Spectrum(data[:,y,x], header, calibration=False, seven_components_fit_authorized=seven_components)
            spec.fit_NII_cube()
            # Numpy arrays are used to facilitate extraction
            line.append([
                spec.get_FWHM_snr_7_components_array(),
                spec.get_amplitude_7_components_array(),
                spec.get_mean_7_components_array()
            ])
        Data_cube.give_update(None, f"NII complete cube fitting progress /{data.shape[1]}")
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
        name: str, default=None. Name of the Map. This is primarily used with the Maps object and it is not necessary to
        provide one to work with this class.
        """
        try:
            self.object = fits_object
            self.data = fits_object.data
            self.header = fits_object.header
        except:
            self.object = fits_object[0]
            self.data = fits_object[0].data
            self.header = fits_object[0].header            
        if name is not None:
            self.name = name

    def __add__(self, other):
        if issubclass(type(other), Map):
            assert self.data.shape == other.data.shape, "Maps of different sizes are being added."
            return Map(fits.PrimaryHDU(self.data + other.data, self.header))
        else:
            return Map(fits.PrimaryHDU(self.data + other, self.header))
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        assert self.data.shape == other.data.shape, "Maps of different sizes are being subtracted."
        return Map(fits.PrimaryHDU(self.data - other.data, self.header))
    
    def __rsub__(self, other):
        return (self.__sub__(other) * -1)

    def __pow__(self, power):
        return Map(fits.PrimaryHDU(self.data ** power, self.header))
    
    def __mul__(self, other):
        if issubclass(type(other), Map):
            assert self.data.shape == other.data.shape, "Maps of different sizes are being multiplied."
            return Map(fits.PrimaryHDU(self.data * other.data, self.header))
        else:
            return Map(fits.PrimaryHDU(self.data * other, self.header))
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if issubclass(type(other), Map):
            assert self.data.shape == other.data.shape, "Maps of different sizes are being divided."
            return Map(fits.PrimaryHDU(self.data / other.data, self.header))
        else:
            return Map(fits.PrimaryHDU(self.data / other, self.header))

    def __rtruediv__(self, other):
        return (self.__truediv__(other))**(-1)

    def __array__(self):
        return self.data
    
    def __eq__(self, other):
        return np.nanmax(np.abs((self.data - other.data) / self.data)) <= 10**(-6) or np.array_equal(self.data, other.data)

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
        nb_pix_bin: int, default=2. Specifies the number of pixels to be binned together along a single axis. For example, the
        default value 2 will give a new map in which every pixel is the mean value of every 4 pixels (2x2 bin).
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
    
    def align_regions(self) -> Map | Map_u:
        """
        Get the squared FWHM map in which the instrumental function has been subtracted with the three regions corrected to fit
        better the WCS.

        Returns
        -------
        Map/Map_u object: global map with the three aligned regions, result of the subtraction of the squared FWHM map and the squared
        instrumental function map.
        """
        regions = [
            pyregion.open("gaussian_fitting/regions/region_1.reg"),
            pyregion.open("gaussian_fitting/regions/region_2.reg"),
            pyregion.open("gaussian_fitting/regions/region_3.reg")
        ]

        # A mask of zeros and ones is created with the regions
        try:
            # If the object is derived from a Map_u object, this statement works
            masks = [region.get_mask(hdu=self.object[0]) for region in regions]
        except:
            # If the object is a Map object, this statement works
            masks = [region.get_mask(hdu=self.object) for region in regions]
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
            specific_maps.append(self.copy())
            specific_maps[-1].header = header

        # Alignment of the specific maps on the global WCS
        reprojected_specific_maps = [specific_map.reproject_on(self) for specific_map in specific_maps]
        # Only the data within the mask is kept
        region_data = [specific_map * masks[i] for i, specific_map in enumerate(reprojected_specific_maps)]
        new_map += region_data[0] + region_data[1] + region_data[2]
        return new_map

    def transfer_temperature_to_FWHM(self, element: str) -> Map:
        """
        Get the FWHM of the thermal Doppler broadening. This is used to convert the temperature map into a FWHM map that
        can be compared with other FWHM maps. This method uses the NII peak's wavelength for the Doppler calculations.

        Arguments
        ---------
        element: str. Name of the element with which the temperature broadening will be calculated. Implemented names are:
        "NII", "Ha", "OIII" and "SII". This makes it so the conversion takes into account the fact that heavier particles
        will be less impacted by high temperatures

        Returns
        -------
        Map object: map of the FWHM due to thermal Doppler broadening.
        """
        elements = {
            "NII":  {"emission_peak": 6583.41, "mass_u": 14.0067},
            "Ha":   {"emission_peak": 6562.78, "mass_u": 1.00784},
            "SII":  {"emission_peak": 6717,    "mass_u": 32.065},
            "OIII": {"emission_peak": 5007,    "mass_u": 15.9994}
        }
        angstroms_center = elements[element]["emission_peak"]     # Emission wavelength of the element
        m = elements[element]["mass_u"] * scipy.constants.u       # Mass of the element
        c = scipy.constants.c                                     # Light speed
        k = scipy.constants.k                                     # Boltzmann constant
        angstroms_FWHM = 2 * float(np.sqrt(2 * np.log(2))) * angstroms_center * (self * k / (c**2 * m))**0.5
        speed_FWHM = c * angstroms_FWHM / angstroms_center / 1000
        return speed_FWHM
    
    def transfer_FWHM_to_temperature(self, element: str) -> Map:
        elements = {
            "NII":  {"emission_peak": 6583.41, "mass_u": 14.0067},
            "Ha":   {"emission_peak": 6562.78, "mass_u": 1.00784},
            "SII":  {"emission_peak": 6717,    "mass_u": 32.065},
            "OIII": {"emission_peak": 5007,    "mass_u": 15.9994}
        }
        angstroms_center = elements[element]["emission_peak"]     # Emission wavelength of the element
        m = elements[element]["mass_u"] * scipy.constants.u       # Mass of the element
        c = scipy.constants.c                                     # Light speed
        k = scipy.constants.k                                     # Boltzmann constant
        angstroms_FWHM = self * 1000 / c * angstroms_center
        temperature = (angstroms_FWHM * c / angstroms_center)**2 * m / (8 * np.log(2) * k)
        return temperature
    
    def get_region_statistics(self, region: pyregion.core.ShapeList=None) -> dict:
        """
        Get the statistics of a region along with a histogram. The supported statistic measures are: median, mean, standard
        deviation, skewness and kurtosis.

        Arguments
        ---------
        region: pyregion.core.ShapeList, default=None. Region in which the statistics need to be calculated. A histogram
        will also be made with the data in this region.
        plot_histogram: bool, default=False. Boolean that specifies if the histogram should be plotted.

        Returns
        -------
        dict: statistics of the region. Every key is a statistic measure.
        """
        # A mask of zeros and ones is created with the region
        if region is None:
            new_map = self.copy()
        else:
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
        return stats
    
    def plot_region_histogram(self, region: pyregion.core.ShapeList=None, title: str=None):
        """
        Plot the histogram of the values in a certain region. If none is provided, then the histogram represents the
        entirety of the Map's data.
        
        Arguments
        ---------
        region: pyregion.core.ShapeList, default=None. Region in which to use the values to plot the histogram. Without
        a region all the data is used.
        title: str, default=None. If present, title of the figure
        """
        if region is None:
            # The NANs are removed from the data from which the statistics are computed
            map_data_without_nan = np.ma.masked_invalid(self.data).compressed()
            plt.hist(map_data_without_nan, bins=np.histogram_bin_edges(map_data_without_nan, bins="fd"))
        else:
            # A mask of zeros and ones is created with the region
            try:
                mask = region.get_mask(hdu=self.object)
            except:
                mask = region.get_mask(hdu=self.object[0])
            mask = np.where(mask == False, np.nan, 1)
            # The map's data is only kept where a mask applies
            new_map = self.copy() * mask
            map_data_without_nan = np.ma.masked_invalid(new_map.data).compressed()
            plt.hist(map_data_without_nan, bins=np.histogram_bin_edges(map_data_without_nan, bins="fd"))
        plt.xlabel("turbulence (km/s)")
        plt.ylabel("nombre de pixels")
        plt.title(title)
        plt.show()



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
        name: str, default=None. Name of the Map. This is primarily used with the Maps object and it is not necessary to
        provide one to work with this class.
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
        if issubclass(type(other), Map_u):
            assert self.data.shape == other.data.shape, "Maps of different sizes are being added."
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data + other.data, self.header),
                                       fits.ImageHDU(self.uncertainties + other.uncertainties, self.header)]))
        else:
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data + other, self.header),
                                       fits.ImageHDU(self.uncertainties, self.header)]))
    
    def __sub__(self, other):
        if issubclass(type(other), Map_u):
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
        if issubclass(type(other), Map):
            assert self.data.shape == other.data.shape, "Maps of different sizes are being multiplied."
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data * other.data, self.header),
                                       fits.ImageHDU((self.uncertainties/self.data + other.uncertainties/other.data)
                                                      * self.data * other.data, self.header)]))
        else:
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data * other, self.header),
                                       fits.ImageHDU(self.uncertainties * other, self.header)]))
        
    def __truediv__(self, other):
        if issubclass(type(other), Map):
            assert self.data.shape == other.data.shape, "Maps of different sizes are being divided."
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data / other.data, self.header),
                                       fits.ImageHDU((self.uncertainties/self.data + other.uncertainties/other.data)
                                                      * self.data / other.data, self.header)]))
        else:
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data / other, self.header),
                                       fits.ImageHDU(self.uncertainties / other, self.header)]))

    def __eq__(self, other):
        return (np.nanmax(np.abs((self.data - other.data) / self.data)) <= 1**(-5) and 
                np.nanmax(np.abs((self.uncertainties - other.uncertainties) / self.uncertainties)) <= 1**(-5))
    
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
        name: str, default=None. Name of the Map. This is primarily used with the Maps object and it is not necessary to
        provide one to work with this class.
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
        return super().__eq__(other) and (np.nanmax(np.abs((self.snr - other.snr) / self.snr)) <= 10**(-6)
                                          or self.snr == other.snr)
    
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
        """
        Initialize a Maps object
        
        Arguments
        ---------
        maps: list. List of maps that may be of any type.
        """
        self.content = {}
        self.names = {}
        for i, individual_map in enumerate(maps):
            self.content[individual_map.name] = individual_map
            self.names[i] = individual_map.name
    
    @classmethod
    def open_from_folder(self, folder_path: str) -> Maps:
        """
        Create a Maps object from the files present in a specific folder. Every file is opened as a map.
        
        Arguments
        ---------
        folder_path: str. Path of the folder in which the maps are contained.
        
        Returns
        -------
        Maps object: the name attribute of each map is the name of the file from which it originated.
        """
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

    def __eq__(self, other):
        for Map in self:
            name = Map.name
            if name not in other.names.values():
                print(f"{name} not present in both Maps.")
            elif self[name].data.shape != other[name].data.shape:
                print(f"{name} does not have the same shape in both Maps.")
            elif self[name] == other[name]:
                print(f"{name} equal.")
            else:
                print(f"{name} not equal.")

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
        """
        Write every map of the Maps object into a folder. This makes use of each map's save_as_fits_file() method.
        
        Arguments
        ---------
        folder_path: str. Indicates the path of the folder in which to create the files.
        """
        for name, element in self.content.items():
            element.save_as_fits_file(f"{folder_path}/{name}.fits")

