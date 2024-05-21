from __future__ import annotations
from astropy.io import fits
from astropy.wcs import WCS
from copy import deepcopy
from eztcolors import Colors as C


class Header(fits.Header):
    """ 
    Encapsulates methods specific to the astropy.io.fits.Header class.
    """
    
    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: Header) -> bool:
        keys_equal = list(self.keys()) == list(other.keys())

        for key, value in self.items():
            if value != other[key]:
                values_equal = False
                break
        else:
            values_equal = True

        return keys_equal and values_equal

    def bin(self, bins: tuple[int, int] | tuple[int, int, int]) -> Header:
        """
        Bins a Header.

        Parameters
        ----------
        bins : tuple[int, int] | tuple[int, int, int]
            Number of pixels to be binned together along each axis (1-3). The size of the tuple varies depending on the
            fits file's number of dimensions. A value of 1 results in the axis not being binned. The axes are in the
            order z, y, x.

        Returns
        -------
        header : Header
            Binned Header.
        """
        assert list(bins) == list(filter(lambda val: val >= 1 and isinstance(val, int), bins)), \
            f"{C.LIGHT_RED}All values in bins must be greater than or equal to 1 and must be integers.{C.END}"
        
        header_copy = self.copy()
        for ax, bin_ in zip(range(1, len(bins) + 1), bins):
            if f"CDELT{4-ax}" in list(self.keys()):
                header_copy[f"CDELT{4-ax}"] *= bin_
            if f"CRPIX{4-ax}" in list(self.keys()):
                header_copy[f"CRPIX{4-ax}"] = (self[f"CRPIX{4-ax}"] - 0.5) / bin_ + 0.5
        
        return header_copy

    def flatten(self, axis: int) -> Header:
        """
        Flattens a Header by removing an axis.

        Parameters
        ----------
        axis : int
            Axis to flatten. Axes are given in their numpy array format, not in the fits header format : axis=0 will
            remove the last header axis.

        Returns
        -------
        header : Header
            Flattened header with the remaining data.
        """
        new_header = self.copy()
        for i in range(axis):
            # Swap axes to place the axis to remove at indice 0 (NAXIS3)
            new_header = new_header.switch_axes(axis - i - 1, axis - i)

        # Erase the axis
        new_header = new_header.remove_axis(0)

        return new_header

    def switch_axes(self, axis_1: int, axis_2: int) -> Header:
        """
        Switches a Header's axes to fit a FitsFile object with swapped axes.
        
        Arguments
        ---------
        axis_1: int
            Source axis, Axes are given in their numpy array format, not in the fits header format : axis=0 will 
            remove the last header axis (NAXIS3).
        axis_2: int
            Destination axis, Axes are given in their numpy array format, not in the fits header format : axis=0 will 
            remove the last header axis (NAXIS3).
        
        Returns
        -------
        header : Header
            Header with the switched axes.
        """
        # Make header readable keywords
        h_axis_1, h_axis_2 = self["NAXIS"] - axis_1, self["NAXIS"] - axis_2

        new_header = self.copy()

        for key in deepcopy(list(self.keys())):
            if key[-1] == str(h_axis_1):
                new_header[f"{key[:-1]}{h_axis_2}-"] = new_header.pop(key)
            elif key[-1] == str(h_axis_2):
                new_header[f"{key[:-1]}{h_axis_1}-"] = new_header.pop(key)
        
        # The modified header keywords are temporarily named with the suffix "-" to prevent duplicates during the
        # process
        # After the process is done, the suffix is removed
        for key in deepcopy(list(new_header.keys())):
            if key[-1] == "-":
                new_header[key[:-1]] = new_header.pop(key)

        return new_header

    def invert_axis(self, axis: int) -> Header:
        """
        Inverts a Header along an axis.

        Parameters
        ----------
        axis : int
            Axis along which the info needs to be inverted.
        
        Returns
        -------
        header : Header
            Header with the inverted axis.
        """
        new_header = self.copy()
        h_axis = self["NAXIS"] - axis
        new_header[f"CDELT{h_axis}"] *= -1
        new_header[f"CRPIX{h_axis}"] = self.data.shape[axis] - self.header[f"CRPIX{h_axis}"] + 1
        return new_header
    
    def crop_axes(self, slices: tuple[slice | int]) -> Header:
        """
        Crops the Header to account for a cropped Cube.

        Parameters
        ----------
        slices : tuple[slice | int]
            Slices to crop each axis. The axes are given in the order z, y, x, which corresponds to axes 3, 2, 1
            respectively.
        
        Returns
        -------
        header : Header
            Cropped Header.
        """
        new_header = self.copy()
        for i, s in enumerate(slices):
            if isinstance(s, slice):
                if s.start is not None:
                    new_header[f"CRPIX{self["NAXIS"]-i}"] -= s.start
                    new_header[f"NAXIS{self["NAXIS"]-i}"] = s.stop - s.start

        return new_header
    
    def remove_axis(self, axis: int) -> Header:
        """
        Removes an axis from a Header.

        Parameters
        ----------
        axis : int
            Axis to remove. Axes are given in their numpy array format, not in the fits header format : axis=0 will
            remove the last header axis (NAXIS3).

        Returns
        -------
        header : Header
            Header with the removed axis.
        """
        new_header = self.copy()
        h_axis = str(self["NAXIS"] - axis)
        for key in deepcopy(list(new_header.keys())):
            if key[-1] == h_axis:
                new_header.pop(key)
        
        new_header["NAXIS"] -= 1

        return new_header
