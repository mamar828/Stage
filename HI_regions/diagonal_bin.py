from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
import scipy

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

    def bin_header(self, nb_pix_bin) -> fits.header:
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
            # A small shift has been empirically corrected
            header_copy["CRPIX1"] += 1.4/3
            header_copy["CRPIX2"] += 1.4/3
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
                answer = "y"
                # answer = input(f"The file '{filename}' already exists, do you wish to overwrite it ? [y/n]")
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
        Initialize an analyzer object. The datacube's file name must be given.

        Arguments
        ---------
        fits_object: astropy.io.fits.hdu.image.PrimaryHDU. Contains the data values and header of the data cube.
        axes_info: tuple of str, default=("longitude l", "latitude b", "spectral axis"). If present, specifies what is
        represented by which axis and the three str then represent the x, y and z axes respectively. This argument is stored
        in the info attribute.
        """
        self.object = fits_object
        self.data = fits_object.data
        self.header = fits_object.header
        self.info = axes_info

    def __str__(self):
        return f"Data_cube shape: {self.data.shape}\nData_cube axes: {list(reversed(self.info.items()))}"

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
                current_shape = data.shape
                if current_shape[1] > current_shape[2]:
                    print(f"Cube to bin will be cut horizontally by {i+1} pixel(s).")
                    data = data[:,:-1,:]
                elif current_shape[1] < current_shape[2]:
                    print(f"Cube to bin will be cut vertically by {i+1} pixel(s).")
                    data = data[:,:,:-1]
                else:
                    print(f"Cube to bin will be cut in both axes by {i+1} pixel(s).")
                    data = data[:,:-1,:-1]
        # The mean value of every pixel group at every channel is calculated and the array returns to a three dimensional state
        return Data_cube(fits.PrimaryHDU(np.nanmean(bin_array, axis=(2,4)), self.bin_header(nb_pix_bin)))

    def rotate(self, angle) -> Data_cube:
        rotated_data = scipy.ndimage.rotate(self.data.swapaxes(0,2).swapaxes(0,1), angle=angle)
        return Data_cube(fits.PrimaryHDU(rotated_data.swapaxes(0,1).swapaxes(0,2), self.header))

    def bin_cube_diagonally(self, nb_pix_bin: int, angle: float) -> Data_cube:
        """
        Bin a Data_cube diagonally. This takes the values of the cube in a diagonal manner.
        """
        # The old shape is stored to make the array reshaping easier
        old_shape = np.array(self.data.shape) / nb_pix_bin
        rotated_data = self.rotate(angle).bin_cube(nb_pix_bin).rotate(-angle).data
        new_shape = np.array(rotated_data.shape)
        # The pixels that need to be cropped are calculated and a slice tuple is made
        crop_pix = np.floor((new_shape[1:] - old_shape[1:]) / 2)
        slices = np.array((crop_pix[0]+1, new_shape[1]-crop_pix[0]-1,crop_pix[1]+1,new_shape[2]-crop_pix[1]-1)).astype(int)
        return Data_cube(fits.PrimaryHDU(rotated_data[:,slices[0]:slices[1],slices[2]:slices[3]], self.bin_header(nb_pix_bin)))
    
    def plot_cube(self):
        plt.imshow(self.data[13,:,:], origin="lower")
        plt.show()

    def switch_axes(self, new_axes: dict) -> Data_cube:
        new_data = np.copy(self.data)
        new_header = self.header.copy()
        old_axes_pos = dict(zip((self.info.values()), (0,1,2)))
        new_axes_pos = dict(zip((new_axes.values()), (0,1,2)))
        dict_keys = list(new_axes_pos.keys())

        if old_axes_pos[dict_keys[0]] == 2:
            new_data = new_data.swapaxes(0,2)
            new_header = self.get_switched_header(0,2)
            if old_axes_pos[dict_keys[1]] == 0:
                new_data = new_data.swapaxes(1,0)
                new_header = self.get_switched_header(1,0)

        elif old_axes_pos[dict_keys[0]] == 1:
            new_data = new_data.swapaxes(0,1)
            new_header = self.get_switched_header(0,1)
            if old_axes_pos[dict_keys[1]] == 2:
                new_data = new_data.swapaxes(0,2)
                new_header = self.get_switched_header(0,2)

        if old_axes_pos[dict_keys[1]] == 2:
            new_data = new_data.swapaxes(1,2)
            new_header = self.get_switched_header(1,2)
        
        return Data_cube(fits.PrimaryHDU(new_data, new_header), new_axes)

    def get_switched_header(self, axis_1, axis_2) -> fits.header:
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



test_data_cube = Data_cube(fits.open("HI_regions/LOOP4_cube_bin2.fits")[0])
# test_data_cube.bin_cube(4).rotate(45).rotate(30).plot_cube()
# test_data_cube.bin_cube_diagonally(4, 30).save_as_fits_file("bin.fits")
# test_data_cube.bin_cube(2).save_as_fits_file("bin.fits")

inverted_data_cube = test_data_cube.switch_axes({"x": "v", "y": "l", "z": "b"})
inverted_data_cube.save_as_fits_file("vlb.fits")
# test_data_cube.save_as_fits_file("test.fits")