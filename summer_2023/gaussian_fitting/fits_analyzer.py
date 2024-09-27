from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
import scipy
import pyregion
import multiprocessing
import time
import os
import uncertainties as unc

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from reproject import reproject_interp
from matplotlib.animation import FuncAnimation

from summer_2023.gaussian_fitting.cube_spectrum import *
from summer_2023.gaussian_fitting.celestial_coords import *


if not sys.warnoptions:
    warnings.simplefilter("ignore")



class Fits_file():
    """
    Encapsulate the methods specific to .fits files that can be used both in data cube and map analysis.
    """

    def bin_header(self, nb_pix_bin: int) -> fits.Header:
        """
        Bin the header to make the WCS match with a binned map.

        Arguments
        ---------
        nb_pix_bin: int. Specifies the number of pixels to be binned together along a single axis.

        Returns
        -------
        astropy.io.fits.Header: binned header.
        """
        header_copy = self.header.copy()
        # The try statement makes it so calibration maps/cubes, which don't have WCS, can also be binned
        try:
            header_copy["CDELT1"] *= nb_pix_bin
            header_copy["CDELT2"] *= nb_pix_bin
            # The CRPIX values correspond to the pixel's center and this must be accounted for when binning
            header_copy["CRPIX1"] = (self.header["CRPIX1"] - 0.5) / nb_pix_bin + 0.5
            header_copy["CRPIX2"] = (self.header["CRPIX2"] - 0.5) / nb_pix_bin + 0.5
        except:
            pass
        return header_copy
    
    def save_as_fits_file(self, filename: str, overwrite: bool=False):
        """
        Write an array as a fits file of the specified name with or without a header. If the object has a header, it 
        will be saved.

        Arguments
        ---------
        filename: str. Indicates the path and name of the created file. If the file already exists, a warning will 
        appear and the file can be overwritten.
        overwrite: bool, default=False. If True, automatically overwrites files without asking for user input.
        """
        # Check if the file already exists
        try:
            fits.open(filename)[0]
            # The file already exists
            while True:
                if overwrite:
                    answer = "y"
                else:
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

    def set_WCS(self, values: dict):
        """
        Set the WCS of a Fits_file object by using RA and DEC objects to modify CRVAL or CDELT values. The CRPIX 
        values can also be modified with floats. If a tuple is offered as value, both axes can be modified at once.

        Arguments
        ---------
        values: dict. Contains the keyword to modify and the value it may take. Supported keywords for tuple assignment
        are "CRPIXS", "CRVALS" and "CDELTS". With these keywords, the value of the first and second axes respectively
        may be given in a tuple.
        """
        for key, value in values.items():
            if isinstance(value, tuple):
                if isinstance(value[0], Celestial_coords):
                    self.header[f"{key[:-1]}1"] = value[0].to_deg()
                    self.header[f"{key[:-1]}2"] = value[1].to_deg()
                else:
                    self.header[f"{key[:-1]}1"] = value[0]
                    self.header[f"{key[:-1]}2"] = value[1]
            else:
                if isinstance(value, Celestial_coords):
                    self.header[key] = value.to_deg()
                else:
                    self.header[key] = value

    def reset_update_file(self):
        """
        Reset the update output file. If the file does not yet exist, it is created. This method should always be 
        called before a loop. This offers feedback when fitting large maps.
        """
        file = open("output.txt", "w")
        file.write("0")
        file.close()

    @staticmethod
    def give_update(info: str):
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
    Encapsulate all the useful methods for the analysis of any data cube.
    """

    def __init__(self, fits_object: fits.PrimaryHDU):
        """
        Initialize a Data_cube object. The data cube's file name must be given.

        Arguments
        ---------
        fits_object: astropy.io.fits.hdu.image.PrimaryHDU. Contains the data values and header of the data cube.
        """
        try:
            self.object = fits_object
            self.data = fits_object.data
            self.header = fits_object.header
        except:
            # Check if the provided argument is a HDUList object
            self.object = fits_object[0]
            self.data = fits_object[0].data
            self.header = fits_object[0].header

    def bin_cube(self, nb_pix_bin: int=2) -> Data_cube:
        """
        Bin a specific cube by the amount of pixels given for every channel. Rectangular cubes will be automatically 
        cropped.

        Arguments
        ---------
        nb_pix_bin: int, default=2. Specifies the number of pixels to be binned together along a single axis. For 
        example, the default value 2 will give a new cube in which every pixel at a specific channel is the mean value 
        of every 4 pixels (2x2 bin) at that same channel.

        Returns
        -------
        Data_cube object: binned cube with a binned header.
        """
        # Calculate the pixels that must be cropped to permit the bin
        cropped_pixels = self.data.shape[1]%nb_pix_bin, self.data.shape[2]%nb_pix_bin
        data = np.copy(self.data)[:, :self.data.shape[1] - cropped_pixels[0], :self.data.shape[2] - cropped_pixels[1]]
        if cropped_pixels[0] != 0:
            print(f"Cube to bin will be cut horizontally by {cropped_pixels[0]} pixel(s).")
        if cropped_pixels[1] != 0:
            print(f"Cube to bin will be cut vertically by {cropped_pixels[1]} pixel(s).")

        # Create a 5 dimensional array that regroups, for every channel, every group of pixels (2 times the nb_pix_bin)
        # into a new grid whose size has been divided by the number of pixels to bin
        bin_array = data.reshape(data.shape[0], int(data.shape[1]/nb_pix_bin), nb_pix_bin,
                                                int(data.shape[2]/nb_pix_bin), nb_pix_bin)

        # The mean value of every pixel group at every channel is calculated and the array returns to a three 
        # dimensional state
        return self.__class__(fits.PrimaryHDU(np.nanmean(bin_array, axis=(2,4)), self.bin_header(nb_pix_bin)))
    
    def plot_cube(self,
            cbar_bounds: tuple=None,
            x_bounds: tuple=None,
            y_bounds: tuple=None,
            title: str=None,
            cbar_label: str=None,
            filename: str=None
        ):
        """
        Plot a Data_cube by animating the change of frames.
        
        Arguments
        ---------
        cbar_bounds: tuple, optional. Indicates the colorbar's bounds if an autoscale is not desired. The tuple's first 
        element is the minimum and the second is the maximum.
        x_bounds: tuple, optional. Sets the plot's horizontal bounds.
        y_bounds: tuple, optional. Sets the plot's vertical bounds.
        title: str, optional. Sets the plot's title.
        cbar_label: str, optional. Sets the colorbar's label.
        filename: str, optional. Sets the filename of the saved figure. If present, the figure will not be plotted but
        only saved.
        """
        fig, ax = plt.subplots()
        if cbar_bounds:
            plot = ax.imshow(self.data[0,:,:], origin="lower", cmap="viridis", vmin=cbar_bounds[0], vmax=cbar_bounds[1])
            # plot = plt.colorbar(plt.imshow(self.data, origin="lower", cmap="viridis", vmin=cbar_bounds[0], vmax=cbar_bounds[1]))
        else:
            plot = ax.imshow(self.data[0,:,:], origin="lower", cmap="viridis")
            # plot = plt.colorbar(plt.imshow(self.data, origin="lower", cmap="viridis"))
        cbar = plt.colorbar(plot)
        cbar.set_label(cbar_label)
        if x_bounds:
            plt.xlim(*x_bounds)
        if y_bounds:
            plt.ylim(*y_bounds)
        plt.title(title)

        def next_slice(frame_number):
            plot.set_array(self.data[frame_number,:,:])
            cbar.update_normal(plot)
        
        animation = FuncAnimation(fig, next_slice, frames=self.data.shape[0], interval=75)

        if filename:
            animation.save(filename, writer="pillow", dpi=300)
        else:
            plt.show()

    def get_header_without_third_dimension(self) -> fits.Header:
        """
        Get the adaptation of a Data_cube object's header for a Map object by removing the spectral axis.

        Returns
        -------
        astropy.io.fits.Header: header with the same but with the third axis removed.
        """
        header = self.header.copy()
        wcs = WCS(header)
        wcs.sip = None
        wcs = wcs.dropaxis(2)
        header = wcs.to_header(relax=True)
        return header
    
    def verify_extract_list(self, extract: list[str]):
        """
        Verify the extract list's format and each element one by one.
        
        Arguments
        ---------
        extract: list of str. Names of the gaussians' parameters to extract. Supported terms are: "mean", "amplitude" 
        and "FWHM". Any combination or number of these terms can be given.
        """
        # The provided extract data is verified before fitting
        assert extract != [], "At least a parameter to be extracted should be provided in the extract list."
        assert isinstance(extract, list), f"Extract argument must be a list, not {type(extract).__name__}."
        for element in extract:
            assert element == "mean" or element == "amplitude" or element == "FWHM", \
                    "Unsupported element in extract list."



class Calibration_data_cube(Data_cube):
    """
    Encapsulate all the methods specific to calibration data cubes.
    """

    def fit(self) -> Map_u:
        """
        Fit the whole data cube and extract the fitted gaussian's FWHM and its uncertainty at every point.
        WARNING: Due to the use of the multiprocessing library, calls to this function NEED to be made inside a 
        condition state with the following phrasing:
        if __name__ == "__main__":
        This prevents the code to recursively create instances of itself that would eventually overload the CPUs.

        Returns
        -------
        Map_u object: map of the FWHM value and its associated uncertainty.
        """
        data = np.copy(self.data)
        pool = multiprocessing.Pool()           # This automatically generates an optimal number of workers
        print(f"Number of processes used: {pool._processes}")
        self.reset_update_file()
        start = time.time()
        fit_fwhm_map = np.array(pool.starmap(worker_split_fit, [(Calibration_data_cube(fits.PrimaryHDU(data[:,y,:],
                                                                                       self.header)), data.shape[1])
                                                                for y in range(data.shape[1])]))
        stop = time.time()
        print("\nFinished in", stop-start, "s.")
        pool.close()
        new_header = self.get_header_without_third_dimension()
        return Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0], new_header),
                                   fits.ImageHDU(fit_fwhm_map[:,:,1], new_header)]))
    
    def get_center_tuple(self) -> tuple[int]:
        """
        Get the coordinates of the calibration's center in the form of a tuple.
        
        Returns
        -------
        tuple of ints: rounded coordinates.
        """
        return round(self.header["CAL_CEN1"]), round(self.header["CAL_CEN2"])
    
    def worker_fit(self, x: int) -> list:
        """
        Fit a pixel of a 2 dimensional Calibration_data_cube, i.e. a Data_cube that is a simple line.

        Arguments
        ---------
        x: int. x coordinate of the pixel that needs to be fitted

        Returns
        -------
        list: fitted gaussian's FWHM value and uncertainty at the pixel. 
        """
        spectrum = Calibration_spectrum(self.data[:,x], self.header)
        spectrum.fit()
        try:
            return spectrum.get_FWHM_speed()
        except:
            return [np.nan, np.nan]



class NII_data_cube(Data_cube):
    """
    Encapsulate all the methods specific to NII data cubes.
    """

    def __init__(self, fits_object: fits.PrimaryHDU, _double_NII_peak_authorized: bool=None):
        """
        Initialize a NII_data_cube object. The data cube's file name must be given.

        Arguments
        ---------
        fits_object: astropy.io.fits.hdu.image.PrimaryHDU. Contains the data values and header of the data cube.
        _double_NII_peak_authorized: bool, default=None. This argument is intended for internal use only. Specifies if 
        double components can be considered when fitting. This is used when fitting the NII_data_cube which sometimes 
        seems to present a multi component peak.
        """
        super().__init__(fits_object)
        self._double_NII_peak_authorized = _double_NII_peak_authorized

    def fit(self, extract: list[str], double_NII_peak_authorized: bool=False) -> list[Maps]:
        """
        Fit the whole data cube to extract the peaks' data. This method presupposes that four OH peaks, one Halpha 
        peak and one NII peak (sometimes two if the seven_components_fits_authorized is set to True) are present.
        WARNING: Due to the use of the multiprocessing library, calls to this function NEED to be made inside a 
        condition state with the following phrasing:
        if __name__ == "__main__":
        This prevents the code to recursively create instances of itself that would eventually overload the CPUs.

        Arguments
        ---------
        extract: list of str. Names of the gaussians' parameters to extract. Supported terms are: "mean", "amplitude" 
        and "FWHM". Any combination or number of these terms can be given.
        double_NII_peak_authorized: bool, default=False. Specifies if double NII peaks are considered possible. If 
        True, the fitting algorithm may detect two components and fit these separately. The NII values in the returned 
        maps for a certain parameter will then be the mean value of the two fitted gaussians at that pixel.

        Returns
        -------
        list of Maps object: the Maps object representing every ray's mean, amplitude or FWHM are returned in the 
        order they were put in argument, thus the list may have a length of 1, 2 or 3. Every Maps object has the maps 
        of every ray present in the provided data cube.
        Note that each map is a Map_usnr object when computing the FWHM whereas the maps are Map_u objects when 
        computing amplitude or mean. In any case, in every Maps object is a Map object having the value 1 when a seven 
        components fit was executed and 0 otherwise.
        """
        self.verify_extract_list(extract)

        data = np.copy(self.data)
        pool = multiprocessing.Pool()           # This automatically generates an optimal number of workers
        print(f"Number of processes used: {pool._processes}")
        self.reset_update_file()
        start = time.time()
        fit_fwhm_map = np.array(pool.starmap(worker_split_fit, [(NII_data_cube(fits.PrimaryHDU(data[:,y,:], 
                                                                        self.header), double_NII_peak_authorized),
                                                                 data.shape[1]) for y in range(data.shape[1])]))
        stop = time.time()
        print("\nFinished in", stop-start, "s.")
        pool.close()
        new_header = self.get_header_without_third_dimension()
        # The fit_fwhm_map has 5 dimensions (x_shape,y_shape,3,7,3) and the last three dimensions are given at every 
        # pixel
        # Third dimension: all three gaussian parameters. 0: fwhm, 1: amplitude, 2: mean
        # Fourth dimension: all rays in the data_cube. 0: OH1, 1: OH2, 2: OH3, 3: OH4, 4: NII, 5: Ha, 6: 7 components 
        # fit map
        # Fifth dimension: 0: data, 1: uncertainties, 2: snr (only when associated to fwhm values)
        # The 7 components fit map is a map taking the value 0 if a single component was fitted onto NII and the value 
        # 1 iftwo components were considered
        fwhm_maps = Maps([
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,0,0], new_header),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,0,1]),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,0,2])]), name="OH1_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,1,0], new_header),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,1,1]),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,1,2])]), name="OH2_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,2,0], new_header),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,2,1]),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,2,2])]), name="OH3_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,3,0], new_header),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,3,1]),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,3,2])]), name="OH4_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,4,0], new_header),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,4,1]),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,4,2])]), name="NII_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,5,0], new_header),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,5,1]),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,5,2])]), name="Ha_fwhm"),
            Map(fits.PrimaryHDU(fit_fwhm_map[:,:,0,6,0], new_header), name="7_components_fit")
        ])
        amplitude_maps = Maps([
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,0,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,0,1])]), name="OH1_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,1,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,1,1])]), name="OH2_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,2,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,2,1])]), name="OH3_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,3,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,3,1])]), name="OH4_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,4,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,4,1])]), name="NII_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,5,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,5,1])]), name="Ha_amplitude"),
            Map(fits.PrimaryHDU(fit_fwhm_map[:,:,1,6,0], new_header), name="7_components_fit")
        ])
        mean_maps = Maps([
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,0,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,0,1])]), name="OH1_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,1,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,1,1])]), name="OH2_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,2,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,2,1])]), name="OH3_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,3,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,3,1])]), name="OH4_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,4,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,4,1])]), name="NII_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,5,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,5,1])]), name="Ha_mean"),
            Map(fits.PrimaryHDU(fit_fwhm_map[:,:,2,6,0], new_header), name="7_components_fit")
        ])
        parameter_names = {"FWHM": fwhm_maps, "amplitude": amplitude_maps, "mean": mean_maps}
        return_list = []
        for element in extract:
            return_list.append(parameter_names[element])
        if len(extract) == 1:
            # If only a single Maps is present, the element itself needs to be returned and not a list of a single 
            # element
            return return_list[-1]
        else:
            return return_list

    def worker_fit(self, x: int) -> list:
        """
        Fit a pixel of a 2 dimensional NII_data_cube, i.e., a Data_cube that is a simple line.

        Arguments
        ---------
        x: int. x coordinate of the pixel that needs to be fitted

        Returns
        -------
        list: FWHM, snr, amplitude and mean value of every fitted gaussians are given along with a map representing if 
        a double NII fit was made. Each coordinate has three sublists. The first has seven values: the first six are 
        the peaks' FWHM with their uncertainty and signal to noise ratio and the last one is a map indicating where  
        fits with seven components were done. The last map outputs 0 for a six components fit and 1 for a seven 
        components fit. The second sublist gives the fitted gaussians amplitude and the third sublist gives their mean 
        value. The two last sublists also have the map that represents if a double NII peak was fitted.
        """
        spectrum = NII_spectrum(self.data[:,x], self.header, 
                                seven_components_fit_authorized=self._double_NII_peak_authorized)
        spectrum.fit()
        if spectrum.is_nicely_fitted_for_NII():
        # if True:
            return [
                spectrum.get_FWHM_snr_7_components_array(), 
                spectrum.get_amplitude_7_components_array(), 
                spectrum.get_mean_7_components_array()
            ]
        else:
            return spectrum.get_list_of_NaN_arrays()



class SII_data_cube(Data_cube):
    """
    Encapsulate all the methods specific to SII data cubes.
    """

    def __init__(self, fits_object: fits.PrimaryHDU, _cube_number: int=None):
        """
        Initialize a NII_data_cube object. The data cube's file name must be given.

        Arguments
        ---------
        fits_object: astropy.io.fits.hdu.image.PrimaryHDU. Contains the data values and header of the data cube.
        _cube_number: int, default=None. This argument is intended for internal use only. Specifies the cube's number 
        (1, 2, 3). This is used when fitting the different SII cubes which do not present the rays at the same spots.
        """
        super().__init__(fits_object)
        self._cube_number = _cube_number

    def fit(self, extract: list[str], cube_number: int) -> list[Maps]:
        """
        Fit the whole data cube to extract the peaks' data. This method presupposes that two OH peaks and two SII 
        peaks are present.
        WARNING: Due to the use of the multiprocessing library, calls to this function NEED to be made inside a 
        condition state with the following phrasing:
        if __name__ == "__main__":
        This prevents the code to recursively create instances of itself that would eventually overload the CPUs.

        Arguments
        ---------
        extract: list of str. Names of the gaussians' parameters to extract. Supported terms are: "mean", "amplitude" 
        and "FWHM". Any combination or number of these terms can be given.
        cube_number: int. Specify the number of the cube being currently fitted.

        Returns
        -------
        list of Maps objects: the Maps object representing every ray's mean, amplitude or FWHM are returned in the 
        order they were put in argument, thus the list may have a length of 1, 2 or 3. Every Maps object has the maps 
        of every ray present in the provided data cube.
        Note that each map is a Map_usnr object when computing the FWHM whereas the maps are Map_u objects when 
        computing amplitude or mean.
        """
        self.verify_extract_list(extract)

        data = np.copy(self.data)
        pool = multiprocessing.Pool()           # This automatically generates an optimal number of workers
        print(f"Number of processes used: {pool._processes}")
        self.reset_update_file()
        start = time.time()
        fit_fwhm_map = np.array(pool.starmap(worker_split_fit, [(SII_data_cube(fits.PrimaryHDU(data[:,y,:], 
                                                                    self.header), cube_number), data.shape[1]) 
                                                                for y in range(data.shape[1])]))
        stop = time.time()
        print("\nFinished in", stop-start, "s.")
        pool.close()
        new_header = self.get_header_without_third_dimension()
        # The fit_fwhm_map has 5 dimensions (x_shape,y_shape,3,4,3) and the last three dimensions are given at every 
        # pixel
        # Third dimension: all three gaussian parameters. 0: fwhm, 1: amplitude, 2: mean
        # Fourth dimension: all rays in the data_cube. 0: OH1, 1: OH2, 2: SII1, 3: SII2
        # Fifth dimension: 0: data, 1: uncertainties, 2: snr (only when associated to fwhm values, False otherwise)
        fwhm_maps = Maps([
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,0,0], new_header),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,0,1]),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,0,2])]), name="OH1_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,1,0], new_header),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,1,1]),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,1,2])]), name="OH2_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,2,0], new_header),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,2,1]),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,2,2])]), name="SII1_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,3,0], new_header),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,3,1]),
                                   fits.ImageHDU(fit_fwhm_map[:,:,0,3,2])]), name="SII2_fwhm")
        ])
        amplitude_maps = Maps([
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,0,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,0,1])]), name="OH1_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,1,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,1,1])]), name="OH2_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,2,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,2,1])]), name="SII1_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,3,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,3,1])]), name="SII2_amplitude")
        ])
        mean_maps = Maps([
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,0,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,0,1])]), name="OH1_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,1,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,1,1])]), name="OH2_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,2,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,2,1])]), name="SII1_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,3,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,3,1])]), name="SII2_mean")
        ])
        parameter_names = {"FWHM": fwhm_maps, "amplitude": amplitude_maps, "mean": mean_maps}
        return_list = []
        for element in extract:
            return_list.append(parameter_names[element])
        if len(extract) == 1:
            # If only a single Maps is present, the element itself needs to be returned and not a list
            return return_list[-1]
        return return_list
    
    def worker_fit(self, x: int) -> list:
        """
        Fit a pixel of a 2 dimensional SII_data_cube, i.e., a Data_cube that is a simple line.

        Arguments
        ---------
        x: int. x coordinate of the pixel that needs to be fitted

        Returns
        -------
        list: FWHM, snr, amplitude and mean value of every fitted gaussians are given for every pixel. Each coordinate 
        has three sublists. The first has four arrays that give the peaks' FWHM with their uncertainty and signal to 
        noise ratio. The two following sublists give every fitted gaussian's amplitude and mean respectively. 
        """
        spectrum = SII_spectrum(self.data[:,x], self.header, self._cube_number)
        try:
            spectrum.fit()
            # Filter the fits depending on their quality
            if spectrum.is_nicely_fitted():
                return [spectrum.get_FWHM_snr_array(), spectrum.get_amplitude_array(), spectrum.get_mean_array()]
            else:
                return spectrum.get_list_of_NaN_arrays()
        except:
            # Sometimes the fit is unsuccessful
            return spectrum.get_list_of_NaN_arrays()



def worker_split_fit(data_cube: Data_cube, number_of_lines: int) -> list:
    """
    Fit an entire line of a Data_cube.

    Arguments
    ---------
    data_cube: Data_cube object. Two dimensional Data_cube that needs to be fitted. All pixels will be fitted one by 
    one.
    number_of_lines: int. Number of lines of the main Data_cube. This is used for giving feedback on the fit's 
    progress.
    
    Returns
    -------
    list: fitted gaussians' values at every pixels along the specified line. For more information on the returned 
    format, see each worker_fit() method directly.
    """
    # Correct for the automatic change of axes when creating a 2D Data_cube
    data_cube.header["NAXIS3"] = data_cube.header["NAXIS2"]
    del data_cube.header["NAXIS2"]

    line = []
    for x in range(data_cube.data.shape[1]):
        line.append(data_cube.worker_fit(x))
    Fits_file.give_update(f"{type(data_cube).__name__} fitting progress /{number_of_lines}")
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
        name: str, default=None. Name of the Map. This is primarily used with the Maps object and it is not necessary 
        to provide one to work with this class.
        """
        try:
            self.object = fits_object
            self.data = fits_object.data
            self.header = fits_object.header
        except:
            # Check if the provided argument is a HDUList object
            self.object = fits_object[0]
            self.data = fits_object[0].data
            self.header = fits_object[0].header
        self.name = name
        self.shape = self.data.shape

    def __add__(self, other):
        if issubclass(type(other), Map):
            assert self.data.shape == other.data.shape, "Maps of different sizes are being added."
            return Map(fits.PrimaryHDU(self.data + other.data, self.header), self.name)
        else:
            return Map(fits.PrimaryHDU(self.data + other, self.header), self.name)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        assert self.data.shape == other.data.shape, "Maps of different sizes are being subtracted."
        return Map(fits.PrimaryHDU(self.data - other.data, self.header), self.name)
    
    def __rsub__(self, other):
        return (self.__sub__(other) * -1)

    def __pow__(self, power):
        return Map(fits.PrimaryHDU(self.data ** power, self.header), self.name)
    
    def __mul__(self, other):
        if issubclass(type(other), Map):
            assert self.data.shape == other.data.shape, "Maps of different sizes are being multiplied."
            return Map(fits.PrimaryHDU(self.data * other.data, self.header), self.name)
        else:
            return Map(fits.PrimaryHDU(self.data * other, self.header), self.name)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if issubclass(type(other), Map):
            assert self.data.shape == other.data.shape, "Maps of different sizes are being divided."
            return Map(fits.PrimaryHDU(self.data / other.data, self.header), self.name)
        else:
            return Map(fits.PrimaryHDU(self.data / other, self.header), self.name)

    def __rtruediv__(self, other):
        return (self.__truediv__(other))**(-1)

    def __array__(self):
        return self.data
    
    def __eq__(self, other):
        return np.nanmax(np.abs((self.data - other.data) / self.data)) <= 10**(-6) or np.array_equal(
                                                                                            self.data, other.data)
    
    def __getitem__(self, key):
        return Map(fits.PrimaryHDU(self.data[key], None), self.name)

    def __len__(self):
        return np.count_nonzero(~np.isnan(self.data))
    
    def copy(self):
        return Map(fits.PrimaryHDU(np.copy(self.data), self.header.copy()), str(self.name))
    
    def add_new_axis(self, new_axis_shape: int) -> Map:
        """
        Reshape a Map object by adding one dimension and filling this axis with preexisting data.

        Arguments
        ---------
        new_axis_shape: int. Desired shape of the new axis that will be appended.

        Returns
        -------
        Map object: new map with its augmented dimension.
        """
        new_data = np.repeat(self.data, new_axis_shape).reshape(*reversed(self.data.shape), new_axis_shape)
        return Map(fits.PrimaryHDU(new_data, self.header), self.name)

    def plot_map(self,
            cbar_bounds: tuple=None,
            x_bounds: tuple=None,
            y_bounds: tuple=None,
            title: str=None,
            cbar_label: str=None,
            filename: str=None
        ):
        """
        Plot the map in matplotlib.pyplot.

        Arguments
        ---------
        cbar_bounds: tuple, optional. Indicates the colorbar's bounds if an autoscale is not desired. The tuple's first 
        element is the minimum and the second is the maximum.
        x_bounds: tuple, optional. Sets the plot's horizontal bounds.
        y_bounds: tuple, optional. Sets the plot's vertical bounds.
        title: str, optional. Sets the plot's title.
        cbar_label: str, optional. Sets the colorbar's label.
        filename: str, optional. Sets the filename of the saved figure. If present, the figure will not be plotted but
        only saved.
        """
        if cbar_bounds:
            plot = plt.colorbar(plt.imshow(self.data, origin="lower", cmap="viridis",
                                           vmin=cbar_bounds[0], vmax=cbar_bounds[1]))
        else:
            plot = plt.colorbar(plt.imshow(self.data, origin="lower", cmap="viridis"))
        if x_bounds:
            plt.xlim(*x_bounds)
        if y_bounds:
            plt.ylim(*y_bounds)
        plt.title(title)
        plot.ax.set_ylabel(cbar_label)
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def plot_two_maps(self, other: Map, cbar_bounds: tuple=None):
        """
        Plot two maps superposed with a certain alpha.

        Arguments
        ---------
        other: Map object. The second map that will be plotted with the magma colormap.
        cbar_bounds: tuple, optional. Indicates the colorbar's bounds if an autoscale is not desired. The tuple's first 
        element is the minimum and the second is the maximum.
        """
        if cbar_bounds is None:
            ax1 = plt.subplot(1,2,1)
            ax2 = plt.subplot(1,2,2)
            plt.colorbar(ax1.imshow(self.data, origin="lower", cmap="viridis"))
            plt.colorbar(ax2.imshow(other.data, origin="lower", cmap="viridis"))
        else:
            ax1 = plt.subplot(1,2,1)
            ax2 = plt.subplot(1,2,2)
            plt.colorbar(ax1.imshow(self.data, origin="lower", cmap="viridis",
                                    vmin=cbar_bounds[0], vmax=cbar_bounds[1]))
            plt.colorbar(ax2.imshow(other.data, origin="lower", cmap="viridis",
                                    vmin=cbar_bounds[0], vmax=cbar_bounds[1]))
        plt.show()

    def bin_map(self, nb_pix_bin: int=2) -> Map:
        """
        Bin the map by the amount of pixels given. Rectangular maps will be automatically cropped.

        Arguments
        ---------
        nb_pix_bin: int, default=2. Specifies the number of pixels to be binned together along a single axis. For 
        example, the default value 2 will give a new map in which every pixel is the mean value of every 4 pixels (2x2 
        bin).

        Returns
        -------
        Map object: binned map.
        """
        cropped_pixels = self.data.shape[0]%nb_pix_bin, self.data.shape[1]%nb_pix_bin
        data = np.copy(self.data)[:self.data.shape[0] - cropped_pixels[0], :self.data.shape[1] - cropped_pixels[1]]
        if cropped_pixels[0] != 0:
            print(f"Map to bin will be cut horizontally by {cropped_pixels[0]} pixel(s).")
        if cropped_pixels[1] != 0:
            print(f"Map to bin will be cut vertically by {cropped_pixels[1]} pixel(s).")

        # Create a 5 dimensional array that regroups, for every channel, every group of pixels (2 times the nb_pix_bin)
        # into a new grid whose size has been divided by the number of pixels to bin
        bin_array = data.reshape(int(data.shape[0]/nb_pix_bin), nb_pix_bin, int(data.shape[1]/nb_pix_bin), nb_pix_bin)
        return Map(fits.PrimaryHDU(np.nanmean(bin_array, axis=(1,3)), self.bin_header(nb_pix_bin)), self.name)
    
    def smooth_order_change(self, center: tuple[int]=(527, 484)) -> Map:
        """
        Smooth the fitted FWHM of the calibration cube for the first two interference order changes. This is needed as 
        the FWHM is reduced at points where the calibration peak changes of interference order. This changes the 
        pixels' value in an order change to the mean value of certain pixels in a 7x7 area around the pixel.

        Arguments
        ---------
        center: tuple of ints, default=(527,484). Specifies the coordinates of the interference pattern's center pixel.

        Returns
        -------
        Map object: FWHM at every point.
        """
        data = np.copy(self.data)
        center = round(center[0]), round(center[1])
        # The bin_factor corrects the distances in the case of a binned array
        bin_factor = center[0] / 527
        # The smoothing_max_thresholds list of ints is defined by trial and error and tunes the pixels to calculate 
        # the mean
        # The first element is used for the first interference order change and the second element is for the second 
        # change
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
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[0]] = np.nan
                    else:
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[1]] = np.nan
                    
                    data[y,x] = np.nanmean(near_pixels)
                    # The addition of near_pixels * 0 makes it so the pixels that have np.nan will not be used
        return Map(fits.PrimaryHDU(data, self.header), self.name)

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
        return Map(fits.PrimaryHDU(reprojection, other.header), self.name)
    
    def align_regions(self) -> Map | Map_u:
        """
        Get the squared FWHM map in which the instrumental function has been subtracted with the three regions 
        corrected to fit better the WCS.

        Returns
        -------
        Map/Map_u object: global map with the three aligned regions, result of the subtraction of the squared FWHM map 
        and the squared instrumental function map.
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

        # Every specific map needs to have the same values than the global map, but the header is changed to fit a 
        # specific region
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
        Get the FWHM of the thermal Doppler broadening. This is used to convert the temperature map into a FWHM map 
        that can be compared with other FWHM maps.

        Arguments
        ---------
        element: str. Name of the element with which the temperature broadening will be calculated. Implemented names 
        are: "NII", "Ha", "OIII" and "SII". This makes it so the conversion takes into account the fact that heavier 
        particles will be less impacted by high temperatures

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
    
    def get_region_statistics(self, region: pyregion.core.ShapeList=None, *, get_new_map: bool=False) -> dict:
        """
        Get the statistics of a region. The supported statistic measures are: median, mean, standard deviation,
        skewness and kurtosis.

        Arguments
        ---------
        region: pyregion.core.ShapeList, default=None. Region in which the statistics need to be calculated. A
        histogram will also be made with the data in this region.
        get_new_map: bool, default=False. If True, also outputs the masked map.

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
        
        stats =  {
            "median": np.nanmedian(new_map.data),
            "mean": np.nanmean(new_map.data),
            "nbpixels": np.count_nonzero(~np.isnan(new_map.data)),
            "standard_deviation": np.nanstd(new_map.data),
            "skewness": scipy.stats.skew(new_map.data, axis=None, nan_policy="omit"),
            "kurtosis": scipy.stats.kurtosis(new_map.data, axis=None, nan_policy="omit")
        }
        if get_new_map:
            return stats, new_map
        else:
            return stats
    
    def plot_region_histogram(self,
            region: pyregion.core.ShapeList=None,
            title: str=None,
            x_label: str=None,
            y_label: str=None,
            bin_width: float=None,
            filename: str=None
        ):
        """
        Plot the histogram of the values in a certain region. If none is provided, then the histogram represents the
        entirety of the Map's data.
        
        Arguments
        ---------
        region: pyregion.core.ShapeList, default=None. Region in which to use the values to plot the histogram. Without
        a region all the data is used.
        title: str, default=None. If present, title of the figure
        x_label: str, default=None. If present, label of the x axis.
        y_label: str, default=None. If present, label of the y axis.
        bin_width: float, default=None. If present, specifies the width of the bins. If absent, the numpy
        histogram_bin_edges function with the fd option will be used instead.
        filename: str, optional. Sets the filename of the saved figure. If present, the figure will not be plotted but
        only saved.
        """
        if not bin_width:
            bins = np.histogram_bin_edges(map_data_without_nan, bins="fd")
        else:
            bins = np.arange(np.nanmin(self.data), np.nanmax(self.data) + bin_width, bin_width)
        if region is None:
            # The nans are removed from the data from which the statistics are computed
            map_data_without_nan = np.ma.masked_invalid(self.data).compressed()
            plt.hist(map_data_without_nan, bins=bins, color="black")
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
            plt.hist(map_data_without_nan, bins=bins, color="black")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def get_cropped_NaNs_array(self) -> np.ndarray:
        # Determine where the data is located
        non_nan_indices = np.where(~np.isnan(self.data))
        # Determine the minimum and maximum indices along each axis
        min_row, max_row = np.min(non_nan_indices[0]), np.max(non_nan_indices[0])
        min_col, max_col = np.min(non_nan_indices[1]), np.max(non_nan_indices[1])
        # Extract the subarray that has been trimmed from the nan ocean
        cropped_array = self[min_row:max_row + 1, min_col:max_col + 1]
        return cropped_array

    def get_autocorrelation_function_array(self, step: float=None) -> np.ndarray:
        cropped_map = self.get_cropped_NaNs_array()
        cropped_array = cropped_map.data

        # Create arrays that will be useful for computing distances
        x, y = np.arange(cropped_array.shape[1]), np.arange(cropped_array.shape[0])
        xx, yy = np.meshgrid(x, y)
        dists_and_multiplications = []

        for y in range(cropped_array.shape[0]):
            if np.nansum(cropped_array[y,:]) == 0.0:        # The row is empty
                continue
            for x in range(cropped_array.shape[1]):
                if not np.isnan(cropped_array[y, x]):
                    multiplication = cropped_array[y, x] * cropped_array
                    dists = np.sqrt((x-xx)**2 + (y-yy)**2)
                    # The multiplication's result is linked to the pixels' distance
                    dists_and_multiplications.append(np.stack((dists, multiplication), axis=2))

        regrouped_dict = self.sort_distances_and_values(dists_and_multiplications, step)

        # The square root of each value is computed first to eliminate all negative data
        # This allows the pixel difference to be considered only once
        mean_values = np.array([(np.append(distance, np.nanmean(array)))
                                for distance, array in regrouped_dict.items()])

        # Extract the x values (distances) and y values (multiplication means divided by the variance)
        y_values = mean_values[:,1] / np.nanvar(cropped_array)
        stacked_values = np.stack((mean_values[:,0], y_values), axis=1)
        sorted_values = stacked_values[np.argsort(stacked_values[:,0])]
        return sorted_values

    def get_structure_function_array(self, step: float=None) -> np.ndarray:
        """ 
        Get the array that represents the structure function.
        WARNING: Due to the use of the multiprocessing library, calls to this function NEED to be made inside a 
        condition state with the following phrasing:
        if __name__ == "__main__":
        This prevents the code to recursively create instances of itself that would eventually overload the CPUs.
        
        Arguments
        ---------
        step: float, default=None. Specifies the bin steps that are to be used to regroup close distances. This helps 
        in smoothing the curve and preventing great bumps with large distances. When equal to None, no bin is made.

        Returns
        -------
        np array: two-dimensional array that contains the function structure (second element on last axis) 
        corresponding to the distance between the pixels (first element on last axis).
        """
        cropped_map = self.get_cropped_NaNs_array()
        cropped_array = cropped_map.data

        # Create arrays that will be useful for computing distances
        x, y = np.arange(cropped_array.shape[1]), np.arange(cropped_array.shape[0])
        xx, yy = np.meshgrid(x, y)
        dists_and_subtractions = []
        
        for y in range(cropped_array.shape[0]):
            if np.nansum(cropped_array[y,:]) == 0.0:        # The row is empty
                continue
            for x in range(cropped_array.shape[1]):
                if not np.isnan(cropped_array[y, x]):
                    subtraction = cropped_array[y, x] - cropped_array
                    dists = np.sqrt((x-xx)**2 + (y-yy)**2)
                    # The subtraction's result is linked to the pixels' distance
                    dists_and_subtractions.append(np.stack((dists, subtraction), axis=2))
        
        regrouped_dict = self.sort_distances_and_values(dists_and_subtractions, step)
        # The square root of each value is computed first to eliminate all negative data
        # This allows the pixel difference to be considered only once
        mean_values = np.array([(np.append(distance, np.nanmean((np.sqrt(array))**4)))
                                           for distance, array in list(regrouped_dict.items())])

        # Extract the x values (distances) and y values (multiplication means divided by the variance)
        y_values = mean_values[:,1] / np.nanvar(cropped_array)
        stacked_values = np.stack((mean_values[:,0], y_values), axis=1)
        sorted_values = stacked_values[np.argsort(stacked_values[:,0])]
        return sorted_values

    def sort_distances_and_values(self, pixel_list, step) -> dict:
        pool = multiprocessing.Pool()
        print(f"Processes used: {pool._processes}")
        start = time.time()

        # Calculate the mean subtraction per distance of every pixel
        dist_and_vals = pool.map(worker_regroup_distances_of_pixel, pixel_list)
        
        group_size = 15
        # Regroup all the subtraction's results per distance of groups of [group_size] pixels, in this case 15
        dist_and_vals_group_1 = pool.map(worker_regroup_pixels,
                                         np.array_split(dist_and_vals, len(dist_and_vals)//group_size))

        # From the already grouped pixels, regroup another [group_size] arrays
        dist_and_vals_group_2 = pool.map(worker_regroup_pixels,
                                         np.array_split(dist_and_vals_group_1, len(dist_and_vals_group_1)//group_size))
        pool.close()

        # Group all the remaining arrays
        dist_and_vals_group_3 = worker_regroup_pixels(dist_and_vals_group_2)
        
        if step is None:
            print("\nAll calculations completed in", time.time() - start, "s.")
            return dist_and_vals_group_3
        else:
            bins = np.arange(0, np.max(list(dist_and_vals_group_3.keys())), step)
            regrouped_dict = {}
            for distance, values in dist_and_vals_group_3.items():
                # Get the closest value to bin to and append the values
                closest_bin = bins[(np.abs(bins-distance)).argmin()]
                regrouped_dict[closest_bin] = np.append(regrouped_dict.get(closest_bin, np.array([])), values)
            print("\nAll calculations completed in", time.time() - start, "s.")
            return regrouped_dict



def worker_regroup_distances_of_pixel(pixel_array: np.ndarray) -> dict:
    """
    Regroup all the values that correspond to the same distance between pixels in the form of a dictionary. Print a 
    "." every time a pixel's analysis has been completed.
    
    Arguments
    ---------
    pixel_array: numpy array. Array whose mean values must be calculated.
    
    Returns
    -------
    dict: the keys are the unique distances between pixels and the values are numpy arrays and correspond to the 
    operation's result from the pixels that are separated by that same distance.
    """
    # The unique distances and their positions in the array are extracted
    unique_distances, indices = np.unique(pixel_array[:,:,0], return_inverse=True)
    dist_and_vals = {}
    for i, unique_distance in enumerate(unique_distances):
        # if unique_distance % 1 == 0:                      # Only get integer distances
        # The indices variable refers to the flattened array
        flat_values = pixel_array[:,:,1].flatten()
        corresponding_values = flat_values[indices == i]
        if corresponding_values[~np.isnan(corresponding_values)].shape != (0,):     # Filter empty arrays
            dist_and_vals[unique_distance] = corresponding_values[~np.isnan(corresponding_values)]
    print(".", end="", flush=True)
    return dist_and_vals

def worker_regroup_pixels(pixel_list: list) -> dict:
    """
    Regroup the dictionary values that are linked to the same key.
    
    Arguments
    ---------
    pixel_list: list of dicts. List of dictionaries in which all the values attributed to the same key will be 
    regrouped.
    
    Returns
    -------
    dict: Single dictionary containing the data of all dictionaries in the pixel_list list of dicts.
    """
    pixel_list_means = {}
    for pixel in pixel_list:
        for key, value in pixel.items():
            if key is not None and value is not None and value.any() and key != 0:
                # Remove empty data and operations calculated with the pixel itself
                # If the key is already present, the new value is appended to the global dict
                pixel_list_means[key] = np.append(pixel_list_means.get(key, np.array([])), value)
    return pixel_list_means



class Map_u(Map):
    """
    Encapsulate the methods specific to maps with uncertainties.
    Note that a Map_u is essentially two Map objects into a single object, the first Map being the data and the second 
    one being its uncertainty. This makes conversion from Map_u -> Map easier via one of the following statements:
    data_map, uncertainty_map = Map_u | uncertainty_map = Map_u[1]
    data_map and uncertainty_map would then be two Map objects.
    It is also possible to create a Map_u object from two Map objects using the from_map_objects method.
    """

    def __init__(self, fits_list, name: str=None):
        """
        Initialize a Map_u object. 

        Arguments
        ---------
        fits_list: astropy.io.fits.hdu.hdulist.HDUList. List of astropy objects. Contains the values, uncertainties 
        and header of the map.
        name: str, default=None. Name of the Map. This is primarily used with the Maps object and it is not necessary 
        to provide one to work with this class.
        """
        self.object = fits_list
        self.data = fits_list[0].data
        self.uncertainties = fits_list[1].data
        self.header = fits_list[0].header
        assert self.data.shape == self.uncertainties.shape, "The data and uncertainties sizes do not match."
        self.name = name
        self.shape = self.data.shape
    
    @classmethod
    def from_Map_objects(cls, map_data: Map, map_uncertainty: Map) -> Map_u:
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
        return cls(fits.HDUList([fits.PrimaryHDU(map_data.data, map_data.header),
                                  fits.ImageHDU(map_uncertainty.data, map_data.header)]))

    def __add__(self, other):
        if issubclass(type(other), Map_u):
            assert self.data.shape == other.data.shape, "Maps of different sizes are being added."
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data + other.data, self.header),
                                    fits.ImageHDU(self.uncertainties + other.uncertainties, self.header)]), self.name)
        else:
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data + other, self.header),
                                       fits.ImageHDU(self.uncertainties, self.header)]), self.name)
    
    def __sub__(self, other):
        if issubclass(type(other), Map_u):
            # assert self.data.shape == other.data.shape, "Maps of different sizes are being subtracted."
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data - other.data, self.header),
                                    fits.ImageHDU(self.uncertainties + other.uncertainties, self.header)]), self.name)
        else:
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data - other, self.header),
                                       fits.ImageHDU(self.uncertainties, self.header)]), self.name)

    def __pow__(self, power):
        return Map_u(fits.HDUList([fits.PrimaryHDU(self.data ** power, self.header),
            fits.ImageHDU(np.abs(self.uncertainties / self.data * power * self.data**power), self.header)]), self.name)
    
    def __mul__(self, other):
        if issubclass(type(other), Map):
            assert self.data.shape == other.data.shape, "Maps of different sizes are being multiplied."
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data * other.data, self.header),
                               fits.ImageHDU((self.uncertainties/self.data + other.uncertainties/other.data, self.name)
                                                      * self.data * other.data, self.header)]))
        else:
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data * other, self.header),
                                       fits.ImageHDU(self.uncertainties * other, self.header)]), self.name)
        
    def __truediv__(self, other):
        if issubclass(type(other), Map):
            assert self.data.shape == other.data.shape, "Maps of different sizes are being divided."
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data / other.data, self.header),
                               fits.ImageHDU((self.uncertainties/self.data + other.uncertainties/other.data, self.name)
                                                      * self.data / other.data, self.header)]))
        else:
            return Map_u(fits.HDUList([fits.PrimaryHDU(self.data / other, self.header),
                                       fits.ImageHDU(self.uncertainties / other, self.header)]), self.name)

    def __eq__(self, other):
        return (np.nanmax(np.abs((self.data - other.data) / self.data)) <= 1**(-5) and 
                np.nanmax(np.abs((self.uncertainties - other.uncertainties) / self.uncertainties)) <= 1**(-5))
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return Map(self.object[key])
        return self.from_Map_objects(Map(fits.PrimaryHDU(self.data, None))[key], 
                                     Map(fits.PrimaryHDU(self.uncertainties, None))[key])
    
    def copy(self):
        return Map_u(fits.HDUList([fits.PrimaryHDU(np.copy(self.data), self.header.copy()),
                                   fits.ImageHDU(np.copy(self.uncertainties), self.header.copy())]), self.name)
    
    def __iter__(self):
        self.n = -1
        return self
    
    def __next__(self):
        self.n += 1
        if self.n > len(self.object) - 1:
            raise StopIteration
        else:
            return Map(fits.PrimaryHDU(self.object[self.n].data, self.header), self.name)
        
    def sum(self):
        return Map_u(fits.HDUList([fits.PrimaryHDU(np.array([np.nansum(self.data)]), None),
                                   fits.ImageHDU(np.array([np.nansum(self.uncertainties)]), None)]))
        
    def to_uncertainties_array(self) -> np.ndarray:
        """
        Transfer a Map_u object to a np.ndarray that uses the uncertainties package.

        Returns
        -------
        np.ndarray: array of the same shape with each element being a uncertainties.ufloat object.
        """
        ufloat_converter = np.vectorize(lambda data, uncertainty: unc.ufloat(data, uncertainty))
        return ufloat_converter(self.data, self.uncertainties)
        
    def add_new_axis(self, new_axis_shape: int) -> Map_u:
        """
        Reshape a Map_u object by adding one dimension and filling this axis with preexisting data.

        Arguments
        ---------
        new_axis_shape: int. Desired shape of the new axis that will be appended.

        Returns
        -------
        Map_u object: new map with its augmented dimension.
        """
        data_map, uncertainty_map = self
        return Map_u.from_Map_objects(data_map.add_new_axis(new_axis_shape),
                                      uncertainty_map.add_new_axis(new_axis_shape))

    def save_as_fits_file(self, filename: str):
        """
        Write the Map_u as a fits file of the specified name with or without a header. If the object has a header, it 
        will be saved. The data uncertainty is saved as the [1] extension of the fits file. To view the uncertainty 
        map on DS9, simply open the file with the following path: File -> Open as -> Multi Extension Cube. The data 
        and its uncertainty will then be visible just like in a data cube.
        
        Arguments
        ---------
        filename: str. Indicates the path and name of the created file. If the file already exists, a warning will 
        appear and the file can be overwritten.
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
        Bin the map by the amount of pixels given. Rectangular maps will be automatically cropped.

        Arguments
        ---------
        nb_pix_bin: int. Specifies the number of pixels to be binned together along a single axis. For example, the 
        default value 2 will give a new map in which every pixel is the mean value of every 4 pixels (2x2 bin).

        Returns
        -------
        Map_u object: binned map.
        """
        data_map, uncertainties_map = self
        return self.from_Map_objects(data_map.bin_map(nb_pix_bin), uncertainties_map.bin_map(nb_pix_bin))
    
    def smooth_order_change(self, center: tuple[int]) -> Map_u:
        """
        Smooth the fitted FWHM of the calibration cube for the first two interference order changes. This is needed as 
        the FWHM is reduced at points where the calibration peak changes of interference order. This changes the 
        pixels' value in an order change to the mean value of certain pixels in a 7x7 area around the pixel.

        Arguments
        ---------
        center: tuple. Specifies the coordinates of the interference pattern's center pixel.

        Returns
        -------
        Map_u object: map with the smoothed instrumental function.
        """
        data = np.copy(self.data)
        uncertainties = np.copy(self.uncertainties)
        # The bin_factor corrects the distances in the case of a binned array
        bin_factor = center[0] / 527
        # The smoothing_max_thresholds list of ints is defined by trial and error and tunes the pixels to calculate 
        # the mean
        # The first element is used for the first interference order change and the second element is for the second
        # change
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
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[0]] = np.nan
                    else:
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[1]] = np.nan
                    
                    data[y,x] = np.nanmean(near_pixels)
                    uncertainties[y,x] = np.nanmean(near_pixels * 0 + near_pixels_uncertainty)
                    # The addition of near_pixels * 0 makes it so the pixels that have np.nan will not be used
        return Map_u(fits.HDUList([fits.PrimaryHDU(data, self.header),
                                   fits.ImageHDU(uncertainties, self.header)]), self.name)

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
        reprojected_data = reproject_interp(self.object[0], other.header, return_footprint=False, 
                                            order="nearest-neighbor")
        reprojected_uncertainties = reproject_interp(self.object[1], other.header, return_footprint=False, 
                                                     order="nearest-neighbor")
        return Map_u(fits.HDUList([fits.PrimaryHDU(reprojected_data, other.header),
                                   fits.ImageHDU(reprojected_uncertainties, self.header)]), self.name)
    
    def get_region_statistics(self, region: pyregion.core.ShapeList=None) -> dict:
        """
        Get the statistics of a region. The supported statistic measures are: median, mean, standard deviation,
        skewness and kurtosis.

        Arguments
        ---------
        region: pyregion.core.ShapeList, default=None. Region in which the statistics need to be calculated. A
        histogram will also be made with the data in this region.

        Returns
        -------
        dict: statistics of the region. Every key is a statistic measure.
        """
        stats, new_map = super().get_region_statistics(region, get_new_map=True)
        # Replace median and mean with values with uncertainties
        median = np.nanmedian(new_map.to_uncertainties_array())
        mean = (new_map.sum()/len(new_map)).to_uncertainties_array()
        stats["median"] = f"{median:.1u}".replace("+/-", " ± ")
        stats["mean"] = f"{mean[0]:.1u}".replace("+/-", " ± ")
        return stats
    
    def get_autocorrelation_function_array(self, step: float=None) -> np.ndarray:
        cropped_map = self.get_cropped_NaNs_array()
        cropped_vals = cropped_map.data
        cropped_uncs = cropped_map.uncertainties

        # Create arrays that will be useful for computing distances
        x, y = np.arange(cropped_vals.shape[1]), np.arange(cropped_vals.shape[0])
        xx, yy = np.meshgrid(x, y)
        dists_and_multiplications = []

        for y in range(cropped_vals.shape[0]):
            if np.nansum(cropped_vals[y,:]) == 0.0:        # The row is empty
                continue
            for x in range(cropped_vals.shape[1]):
                if not np.isnan(cropped_vals[y,x]):
                    multiplication_val = cropped_vals[y,x] * cropped_vals
                    multiplication_unc = ((cropped_uncs[y,x]/cropped_vals[y,x] + cropped_uncs/cropped_vals)
                                          * multiplication_val)
                    dists = np.sqrt((x-xx)**2 + (y-yy)**2)
                    # The multiplication's result is linked to the pixels' distance
                    dists_and_multiplications.append(np.stack((dists, multiplication_val, multiplication_unc), axis=2))
        
        regrouped_dict = self.sort_distances_and_values(dists_and_multiplications, step)
        
        # The square root of each value is computed first to eliminate all negative data
        # This allows the pixel difference to be considered only once
        mean_values = np.array([(np.append(distance, np.nanmean(array, axis=0))) 
                                for distance, array in regrouped_dict.items()])

        # Extract the x values (distances) and y values (multiplication means divided by the variance)
        y_values = mean_values[:,1:] / np.nanvar(cropped_vals)
        stacked_values = np.stack((mean_values[:,0], y_values[:,0], y_values[:,1]), axis=1)
        sorted_values = stacked_values[np.argsort(stacked_values[:,0])]
        return sorted_values

    def get_structure_function_array(self, step: float=None) -> np.ndarray:
        """ 
        Get the array that represents the structure function.
        WARNING: Due to the use of the multiprocessing library, calls to this function NEED to be made inside a 
        condition state with the following phrasing:
        if __name__ == "__main__":
        This prevents the code to recursively create instances of itself that would eventually overload the CPUs.
        
        Arguments
        ---------
        step: float, default=None. Specifies the bin steps that are to be used to regroup close distances. This helps 
        in smoothing the curve and preventing great bumps with large distances. When equal to None, no bin is made.

        Returns
        -------
        np array: two-dimensional array that contains the function structure (second element on last axis) 
        corresponding to the distance between the pixels (first element on last axis).
        """
        cropped_map = self.get_cropped_NaNs_array()
        cropped_vals = cropped_map.data
        cropped_uncs = cropped_map.uncertainties

        # Create arrays that will be useful for computing distances
        x, y = np.arange(cropped_vals.shape[1]), np.arange(cropped_vals.shape[0])
        xx, yy = np.meshgrid(x, y)
        dists_and_subtractions = []

        for y in range(cropped_vals.shape[0]):
            if np.nansum(cropped_vals[y,:]) == 0.0:        # The row is empty
                continue
            for x in range(cropped_vals.shape[1]):
                if not np.isnan(cropped_vals[y,x]):
                    subtraction_val = cropped_vals[y,x] - cropped_vals
                    subtraction_unc = cropped_uncs[y,x] + cropped_uncs
                    dists = np.sqrt((x-xx)**2 + (y-yy)**2)
                    # The multiplication's result is linked to the pixels' distance
                    dists_and_subtractions.append(np.stack((dists, subtraction_val, subtraction_unc), axis=2))
        
        regrouped_dict = self.sort_distances_and_values(dists_and_subtractions, step)

        # The square root of each value is computed first to eliminate all negative data
        # This allows the pixel difference to be considered only once
        mean_values = np.array([(np.append(distance, np.nanmean((np.sqrt(array))**4, axis=0))) 
                                 for distance, array in list(regrouped_dict.items())])

        # Extract the x values (distances) and y values (subtraction means divided by the variance, squared)
        y_values = mean_values[:,1:] / np.nanvar(cropped_vals)
        stacked_values = np.stack((mean_values[:,0], y_values[:,0], y_values[:,1]), axis=1)
        sorted_values = stacked_values[np.argsort(stacked_values[:,0])]
        return sorted_values

    def sort_distances_and_values(self, pixel_list, step) -> dict:
        pool = multiprocessing.Pool()
        print(f"Processes used: {pool._processes}")
        start = time.time()

        # Calculate the mean subtraction per distance of every pixel
        dist_and_vals = pool.map(worker_regroup_distances_of_pixel_u, pixel_list)

        group_size = 15
        # Regroup all the subtraction's results per distance of groups of [group_size] pixels, in this case 15
        dist_and_vals_group_1 = pool.map(worker_regroup_pixels_u,
                                         np.array_split(dist_and_vals, len(dist_and_vals)//group_size))

        # From the already grouped pixels, regroup another [group_size] arrays
        dist_and_vals_group_2 = pool.map(worker_regroup_pixels_u,
                                         np.array_split(dist_and_vals_group_1, len(dist_and_vals_group_1)//group_size))
        pool.close()

        # Group all the remaining arrays
        dist_and_vals_group_3 = worker_regroup_pixels_u(dist_and_vals_group_2)

        if step is None:
            print("\nAll calculations completed in", time.time() - start, "s.")
            return dist_and_vals_group_3
        else:
            bins = np.round(np.arange(0, np.max(list(dist_and_vals_group_3.keys())), step),1)
            regrouped_dict = {}
            for distance, values_and_uncert in dist_and_vals_group_3.items():
                # Get the closest value to bin to and append the values
                closest_bin = bins[(np.abs(bins-distance)).argmin()]

                preexisting_data = regrouped_dict.get(closest_bin, None)
                if preexisting_data is not None:
                    regrouped_dict[closest_bin] = np.vstack((preexisting_data, values_and_uncert))
                else:
                    regrouped_dict[closest_bin] = values_and_uncert
                
            print("\nAll calculations completed in", time.time() - start, "s.")
            return regrouped_dict



def worker_regroup_distances_of_pixel_u(pixel_array: np.ndarray) -> dict:
    """
    Regroup all the values that correspond to the same distance between pixels in the form of a dictionary. Print a 
    "." every time a pixel's analysis has been completed.
    
    Arguments
    ---------
    pixel_array: numpy array. Array whose mean values must be calculated.
    
    Returns
    -------
    dict: the keys are the unique distances between pixels and the values are numpy arrays and correspond to the 
    operation's result from the pixels that are separated by that same distance.
    """
    # The unique distances and their positions in the array are extracted
    unique_distances, indices = np.unique(pixel_array[:,:,0], return_inverse=True)
    dist_and_vals = {}
    for i, unique_distance in enumerate(unique_distances):
        # if unique_distance % 1 == 0:                      # Only get integer distances
        # The indices variable refers to the flattened array
        flat_values = pixel_array[:,:,1].flatten()
        corresponding_values = flat_values[indices == i]
        if corresponding_values[~np.isnan(corresponding_values)].shape != (0,):     # Filter empty arrays
            flat_uncertainties = pixel_array[:,:,2].flatten()
            corresponding_uncertainties = flat_uncertainties[indices == i]
            dist_and_vals[unique_distance] = np.transpose(
                np.array((corresponding_values[~np.isnan(corresponding_values)],
                          corresponding_uncertainties[~np.isnan(corresponding_uncertainties)])))
    print(".", end="", flush=True)
    return dist_and_vals

def worker_regroup_pixels_u(pixel_list: list) -> dict:
    """
    Regroup the dictionary values that are linked to the same key.
    
    Arguments
    ---------
    pixel_list: list of dicts. List of dictionaries in which all the values attributed to the same key will be 
    regrouped.
    
    Returns
    -------
    dict: Single dictionary containing the data of all dictionaries in the pixel_list list of dicts.
    """
    pixel_list_means = {}
    for pixel in pixel_list:
        for key, value_and_uncert in pixel.items():
            if key is not None and value_and_uncert[0] is not None and value_and_uncert[0].any() and key != 0:
                # Remove empty data and operations calculated with the pixel itself (when the distance equals 0)
                # If the key is already present, the new value is appended to the global dict
                preexisting_means = pixel_list_means.get(key, None)
                if preexisting_means is not None:
                    pixel_list_means[key] = np.vstack((preexisting_means, value_and_uncert))
                else:
                    pixel_list_means[key] = value_and_uncert
    return pixel_list_means



class Map_usnr(Map_u):
    """
    Encapsulate the methods specific to maps with uncertainties and signal to noise ratios.
    Note that a Map_usnr is essentially three Map objects into a single object, the first Map being the data, the 
    second one being its uncertainty and the third one being the signal to noise ratio. This makes conversion from 
    Map_usnr -> Map easier
    via one of the following statements: data_map, uncertainty_map, snr_map = Map_usnr | snr_map = Map_usnr[2]
    data_map, uncertainty_map and snr_map would then be three Map objects.
    It is also possible to create a Map_usnr object from three Map objects using the from_map_objects method.
    """
    
    def __init__(self, fits_list, name: str=None):
        """
        Initialize a Map_usnr object. 

        Arguments
        ---------
        fits_list: astropy.io.fits.hdu.hdulist.HDUList. List of astropy objects. Contains the values, uncertainties, 
        signal to noise ratio and header of the map.
        name: str, default=None. Name of the Map. This is primarily used with the Maps object and it is not necessary 
        to provide one to work with this class.
        """
        super().__init__(fits_list, name)
        self.snr = fits_list[2].data
        assert self.data.shape == self.snr.shape, "The data and signal to noise ratios sizes do not match."

    @classmethod
    def from_Map_objects(cls, map_data: Map, map_uncertainty: Map, map_snr: Map) -> Map_usnr:
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
        return cls(fits.HDUList([fits.PrimaryHDU(map_data.data, map_data.header),
                                  fits.ImageHDU(map_uncertainty.data, map_data.header),
                                  fits.ImageHDU(map_snr.data, map_data.header)]))
    
    @classmethod
    def from_Map_u_object(cls, map_values: Map_u, map_snr: Map) -> Map_usnr:
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
        return cls(fits.HDUList([fits.PrimaryHDU(map_values.data, map_values.header),
                                  fits.ImageHDU(map_values.uncertainties, map_values.header),
                                  fits.ImageHDU(map_snr.data, map_values.header)]))
    
    def __eq__(self, other):
        return super().__eq__(other) and (np.nanmax(np.abs((self.snr - other.snr) / self.snr)) <= 10**(-6)
                                          or self.snr == other.snr)

    def __getitem__(self, key):
        if isinstance(key, int):
            return Map(self.object[key])
        return self.from_Map_objects(Map(fits.PrimaryHDU(self.data, None))[key],
                                     Map(fits.PrimaryHDU(self.uncertainties, None))[key],
                                     Map(fits.PrimaryHDU(self.snr, None))[key])
    
    def copy(self):
        return self.from_Map_u_object(super().copy(), np.copy(self.snr))
    
    def save_as_fits_file(self, filename: str):
        """
        Write the Map_usnr as a fits file of the specified name with or without a header. If the object has a header, 
        it will be saved. The data uncertainty is saved as the [1] extension of the fits file and the signal to noise 
        ratio is saved as the [2] extension of the. To view the uncertainty and signal to noise ratio maps on DS9, 
        simply open the file with the following path: File -> Open as -> Multi Extension Cube. The data, its 
        uncertainty and its snr will then be visible just like in a data cube.
        
        Arguments
        ---------
        filename: str. Indicates the path and name of the created file. If the file already exists, a warning will 
        appear and the file can be overwritten.
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
        Filter the map's values by keeping only the data that has a signal to noise ratio superior or equal to the 
        provided quantity.

        Arguments
        ---------
        snr_threshold: float, default=6. The pixels that have a signal to noise ratio inferior to this quantity will be
        excluded.

        Returns
        -------
        Map_usnr object: map with the filtered data.
        """
        mask = np.ma.masked_less(self.snr, snr_threshold).mask
        mask = np.where(mask == True, np.nan, 1)
        return Map_usnr(fits.HDUList([fits.PrimaryHDU(self.data * mask, self.header),
                                      fits.ImageHDU(self.uncertainties * mask, self.header),
                                      fits.ImageHDU(self.snr * mask, self.header)]), self.name)



class Maps():
    """
    Encapsulates the methods that are specific to a multitude of linked maps. This class is mainly used as the output 
    of the fit() method and allows for many convenient operations. This class can be seen as an analoguous to the dict 
    class.
    """

    def __init__(self, maps: list[Map]):
        """
        Initialize a Maps object.
        
        Arguments
        ---------
        maps: list of Map objects. List of maps that may be of any type.
        """
        self.content = {}
        self.names = {}
        for i, individual_map in enumerate(maps):
            self.content[individual_map.name] = individual_map
            self.names[i] = individual_map.name
    
    @classmethod
    def open_from_folder(cls, folder_path: str) -> Maps:
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
                        try:
                            maps.append(Map(fits.open(f"{folder_path}/{file}")[0], name=name))
                        except:
                            print(f"File {file} could not be opened.")
        return cls(maps)
    
    @classmethod
    def build_from_dict(cls, dictio: dict) -> Maps:
        """
        Create a Maps object from a dict object. A copy of each map is used.
        
        Arguments
        ---------
        dictio: dict. Each key corresponds to the map's name and each value to the map object itself.
        
        Returns
        -------
        Maps object: newly created object."""
        maps = []
        for name, individual_map in dictio.items():
            map_copy = individual_map.copy()
            map_copy.name = name
            maps.append(map_copy)
        return cls(maps)
    
    def __str__(self):
        return "\n".join(list(self.content.keys()))

    def __eq__(self, other):
        return_str = ""
        for Map in self:
            name = Map.name
            if name not in other.names.values():            # ANSI color codes are used to integrate colors
                return_str += f"\033[1;33;40m{name} not present in both Maps.\n"
            elif self[name].data.shape != other[name].data.shape:
                return_str += f"\033[1;38;5;208m{name} does not have the same shape in both Maps.\n"
            elif self[name] == other[name]:
                return_str += f"\033[1;32;40m{name} equal.\n"
            else:
                return_str += f"\033[1;31;40m{name} not equal.\n"
        return return_str[:-2] + "\033[0m"          # Remove trailing "\n"

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
    
    def __setitem__(self, map_name, map_object):
        self.content[map_name] = map_object
        self.names[len(self.names)] = map_name
    
    def save_as_fits_file(self, folder_path: str):
        """
        Write every map of the Maps object into a folder. This makes use of each map's save_as_fits_file() method.
        
        Arguments
        ---------
        folder_path: str. Indicates the path of the folder in which to create the files. If the folder does not exist, 
        it is created.
        """
        if not os.path.exists(folder_path):
            # Create folder if necessary
            os.makedirs(folder_path)

        for name, element in self.content.items():
            element.save_as_fits_file(f"{folder_path}/{name}.fits")
