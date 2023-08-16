from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
import scipy
import astropy

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from PIL import Image
from copy import deepcopy

if not sys.warnoptions:
    warnings.simplefilter("ignore")



class Fits_file():
    """
    Encapsulate the methods specific to .fits files that can be used both in data cube and map analysis.
    """

    def bin_header(self, nb_pix_bin: int) -> fits.Header:
        """
        Bin the header to make the WCS match with a binned map.
        Note that this method is more advanced than the one present in gaussian_fitting/fits_analyzer as this one is able
        to correct for the microscopic distortions that affect the WCS. These distortions are better perceived when reducing
        considerably the map's size.

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
    
    def get_header_in_equatorial_coords(self):
        # Convert CRVALs and CDELTs in equatorial coordinates
        galactic_CRVALS = astropy.coordinates.SkyCoord(l=self.header["CRVAL1"]*u.degree, 
                                                       b=self.header["CRVAL2"]*u.degree, frame="galactic")
        galactic_CDELTS = astropy.coordinates.SkyCoord(l=self.header["CDELT1"]*u.degree, 
                                                       b=self.header["CDELT2"]*u.degree, frame="galactic")
        intermediate_CRVALS, intermediate_CDELTS = galactic_CRVALS.transform_to("icrs"), galactic_CDELTS.transform_to("icrs")
        equatorial_CRVALS, equatorial_CDELTS = intermediate_CRVALS.transform_to("fk5"), intermediate_CDELTS.transform_to("fk5")

        new_header = self.header.copy()
        new_header["CRVAL1"], new_header["CRVAL2"] = equatorial_CRVALS.ra.deg, equatorial_CRVALS.dec.deg
        new_header["CDELT1"], new_header["CDELT2"] = equatorial_CDELTS.ra.deg, equatorial_CDELTS.dec.deg
        new_header["CTYPE1"], new_header["CTYPE2"] = "RA---TAN", "DEC--TAN"
        return new_header
    
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
                # answer = "y"
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



class Data_cube(Fits_file):
    """
    Encapsulate all the useful methods for the analysis of a data cube.
    """

    def __init__(self, fits_object: fits.PrimaryHDU, axes_info: dict={"x": "l", "y": "b", "z": "v"}):
        """
        Initialize a Data_cube object. The data cube's PrimaryHDU must be given.

        Arguments
        ---------
        fits_object: astropy.io.fits.hdu.image.PrimaryHDU. Contains the data values and header of the data cube.
        axes_info: dict, default={"x": "l", "y": "b", "z": "v"}. Specifies what is represented by which axis and it is mainly
        used in the swap_axes() method. The given dict is stored in the info attribute and information on a Data_cube's axes can
        always be found by printing said Data_cube.
        """
        try:
            self.object = fits_object
            self.data = fits_object.data
            self.header = fits_object.header
        except:
            self.object = fits_object[0]
            self.data = fits_object[0].data
            self.header = fits_object[0].header
        self.info = axes_info

    def __str__(self):
        return f"Data_cube shape: {self.data.shape}\nData_cube axes: {list(reversed(self.info.items()))}"

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
    
    def bin_cube(self, nb_pix_bin: int=2) -> Data_cube:
        """
        Bin a specific cube by the amount of pixels given for every channel.

        Arguments
        ---------
        nb_pix_bin: int, default=2. Specifies the number of pixels to be binned together along a single axis. For example, the default
        value 2 will give a new cube in which every pixel at a specific channel is the mean value of every 4 pixels (2x2 bin) at that
        same channel.

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
        
        return Data_cube(fits.PrimaryHDU(np.nanmean(bin_array, axis=(2,4)), self.bin_header(nb_pix_bin)))

    def plot_cube(self):
        plt.imshow(self.data[13,:,:], origin="lower")
        plt.show()

    def swap_axes(self, new_axes: dict) -> Data_cube:
        """
        Swap a Data_cube's axes to observe different correlations.
        
        Arguments
        ---------
        new_axes: dict. Each axis needs to be attributed a certain type previously specified in the axes_info.
        
        Returns
        -------
        Data_cube object: newly swapped Data_cube.
        """
        new_data = np.copy(self.data)
        new_header = self.header.copy()
        old_axes_pos = dict(zip((self.info.values()), (0,1,2)))
        new_axes_pos = dict(zip((new_axes.values()), (0,1,2)))
        dict_keys = list(new_axes_pos.keys())

        if old_axes_pos[dict_keys[0]] == 2:                         # Check if the new first axis was the old last axis
            new_data = new_data.swapaxes(0,2)
            new_header = self.get_switched_header(0,2)
            if old_axes_pos[dict_keys[1]] == 0:
                new_data = new_data.swapaxes(1,0)
                new_header = self.get_switched_header(1,0)

        elif old_axes_pos[dict_keys[0]] == 1:                       # Check if the new first axis was the old second axis
            new_data = new_data.swapaxes(0,1)
            new_header = self.get_switched_header(0,1)
            if old_axes_pos[dict_keys[1]] == 2:
                new_data = new_data.swapaxes(0,2)
                new_header = self.get_switched_header(0,2)

        if old_axes_pos[dict_keys[1]] == 2:                         # Check if the new second axis was the old last axis
            new_data = new_data.swapaxes(1,2)
            new_header = self.get_switched_header(1,2)
        
        return Data_cube(fits.PrimaryHDU(new_data, new_header), new_axes)

    def get_switched_header(self, axis_1: int, axis_2: int) -> fits.Header:
        """
        Get the astropy header with switched axes to fit a Data_cube whose axes were also swapped.
        
        Arguments
        ---------
        axis_1: int. Source axis.
        axis_2: int. Destination axis.
        
        Returns
        -------
        astropy.io.fits.Header: header copy with switched axes.
        """
        new_header = self.header.copy()
        h_axis_1, h_axis_2 = axis_1+1, axis_2+1                             # The header uses 1-based indexing
        
        for header_element in deepcopy(list(new_header.keys())):
            if header_element[-1] == str(h_axis_1):
                new_header[f"{header_element[:-1]}{h_axis_2}-"] = new_header.pop(header_element)
            elif header_element[-1] == str(h_axis_2):
                new_header[f"{header_element[:-1]}{h_axis_1}-"] = new_header.pop(header_element)
        
        # The modified header keywords are temporarily named with the suffix "-" to prevent duplicates during the process
        # After the process is done, the suffix is removed
        for header_element in deepcopy(list(new_header.keys())):
            if header_element[-1] == "-":
                new_header[header_element[:-1]] = new_header.pop(header_element)
        
        return new_header



HI_data = Data_cube(fits.open("HI_regions/LOOP4_cube.fits")[0], axes_info={"x": "l", "y": "b", "z": "v"})
print(HI_data)
vlb = HI_data.bin_cube(2).swap_axes({"x": "v", "y": "l", "z": "b"})
print(vlb)
vlb.save_as_fits_file("vlb.fits")

