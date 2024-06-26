from __future__ import annotations
from astropy.io import fits
from copy import deepcopy
from eztcolors import Colors as C


class Header(fits.Header):
    """ 
    Encapsulates methods specific to the astropy.io.fits.Header class.
    Note : the axes are always given in their numpy array format, not in the fits header format. For example, axis=0
    targets the first numpy array axis, and therefore the last header axis (3). Values of 0,1,2 target respectively
    z, y and x.
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
    
    def h_axis(self, axis: int) -> int:
        """
        Converts a numpy axis to a header axis.

        Parameters
        ----------
        axis : int
            Axis to convert to a header axis.

        Returns
        -------
        header axis : int
            Axis converted to a header axis.
        """
        h_axis = self["NAXIS"] - axis
        return h_axis

    def bin(self, bins: list[int] | tuple[int, int] | tuple[int, int, int]) -> Header:
        """
        Bins a Header.

        Parameters
        ----------
        bins : list[int] | tuple[int, int] | tuple[int, int, int]
            Number of pixels to be binned together along each axis (1-3). The size of the tuple varies depending on the
            fits file's number of dimensions. A value of 1 results in the axis not being binned. Read the note in the
            declaration of this function to properly indicate the axes.

        Returns
        -------
        header : Header
            Binned Header.
        """
        assert list(bins) == list(filter(lambda val: val >= 1 and isinstance(val, int), bins)), \
            f"{C.LIGHT_RED}All values in bins must be greater than or equal to 1 and must be integers.{C.END}"
        
        header_copy = self.copy()
        for ax, bin_ in enumerate(bins):
            h_ax = self.h_axis(ax)
            if f"CDELT{h_ax}" in list(self.keys()):
                header_copy[f"CDELT{h_ax}"] *= bin_
            if f"CRPIX{h_ax}" in list(self.keys()):
                header_copy[f"CRPIX{h_ax}"] = (self[f"CRPIX{h_ax}"] - 0.5) / bin_ + 0.5
        
        return header_copy

    def flatten(self, axis: int) -> Header:
        """
        Flattens a Header by removing an axis.

        Parameters
        ----------
        axis : int
            Axis to flatten.

        Returns
        -------
        header : Header
            Flattened header with the remaining data.
        """
        new_header = self.copy()
        for i in range(axis):
            # Swap axes to place the axis to remove at indice 0 (NAXIS3)
            new_header = new_header.swap_axes(axis - i - 1, axis - i)

        # Erase the axis
        new_header = new_header.remove_axis(0)

        return new_header

    def swap_axes(self, axis_1: int, axis_2: int) -> Header:
        """
        Switches a Header's axes to fit a FitsFile object with swapped axes.
        
        Parameters
        ----------
        axis_1: int
            Source axis.
        axis_2: int
            Destination axis.
        
        Returns
        -------
        header : Header
            Header with the switched axes.
        """
        # Make header readable keywords
        h_axis_1, h_axis_2 = self.h_axis(axis_1), self.h_axis(axis_2)
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
        h_axis = self.h_axis(axis)
        new_header[f"CDELT{h_axis}"] *= -1
        new_header[f"CRPIX{h_axis}"] = self[f"NAXIS{h_axis}"] - self[f"CRPIX{h_axis}"] + 1
        return new_header
    
    def crop_axes(self, slices: tuple[slice | int]) -> Header:
        """
        Crops the Header to account for a cropped FitsFile.

        Parameters
        ----------
        slices : tuple[slice | int]
            Slices to crop each axis. An integer slice will not crop the axis.
        
        Returns
        -------
        header : Header
            Cropped Header.
        """
        new_header = self.copy()
        for i, s in enumerate(slices):
            if isinstance(s, slice):
                h_axis = self.h_axis(i)
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else self[f"NAXIS{h_axis}"]
                new_header[f"CRPIX{h_axis}"] -= start
                new_header[f"NAXIS{h_axis}"] = stop - start

        return new_header
    
    def remove_axis(self, axis: int) -> Header:
        """
        Removes an axis from a Header.

        Parameters
        ----------
        axis : int
            Axis to remove.

        Returns
        -------
        header : Header
            Header with the removed axis.
        """
        new_header = self.copy()
        h_axis = str(self.h_axis(axis))
        for key in deepcopy(list(new_header.keys())):
            if key[-1] == h_axis:
                new_header.pop(key)
        
        new_header["NAXIS"] -= 1

        return new_header
    
    def concatenate(self, other: Header, axis: int) -> Header:
        """
        Concatenates two headers along an axis. The Header closest to the origin should be the one to call this method.
        This method is used if a FitsFile whose header was previously cropped (with Header.crop_axes) needs to be
        re-concatenated. The FitsFiles are considered directly next to each other.

        Parameters
        ----------
        other : Header
            Second Header to merge the current Header with.
        axis : int
            Index of the axis on which to execute the merge.
        
        Returns
        -------
        header : Header
            Concatenated Header.
        """
        new_header = self.copy()
        h_axis = self.h_axis(axis)
        new_header[f"NAXIS{h_axis}"] += other[f"NAXIS{h_axis}"]
        return new_header

    def get_frame(self, value: float, axis: int=0) -> int:
        """
        Gives the number of the frame closest to the specified value, along the given axis.
        
        Parameters
        ----------
        value : float
            Value to determine the frame. This can be a value in the range of any axis.
        axis : int, default=0
            Axis along which to get the frame. The default axis (0) gives the frame along a cube's spectral axis.

        Returns
        -------
        frame : int
            Number of the frame closest to the specified value.
        """
        h_axis = self.h_axis(axis)
        frame_number = (value - self[f"CRVAL{h_axis}"]) / self[f"CDELT{h_axis}"] + self[f"CRPIX{h_axis}"]
        rounded_frame = round(frame_number)
        return rounded_frame

    def get_value(self, frame: int, axis: int=0) -> float:
        """
        Gives the value associated with the specified frame, along the given axis.
        
        Parameters
        ----------
        frame : int
            Frame to determine the value. This should be a frame in the range of any axis.
        axis : int, default=0
            Axis along which to get the value at the specified frame. The default axis (0) gives the value along a
            cube's spectral axis.

        Returns
        -------
        value : float
            Value at the given frame.
        """
        h_axis = self.h_axis(axis)
        value = (frame - self[f"CRPIX{h_axis}"]) * self[f"CDELT{h_axis}"] + self[f"CRVAL{h_axis}"]
        return value
