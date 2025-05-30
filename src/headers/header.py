from __future__ import annotations
from astropy.io import fits
from astropy.wcs import WCS
from copy import deepcopy
from numpy import cos, radians
from typing import Self
from colorist import BrightColor as C

from src.hdu.fits_file import FitsFile


class Header(fits.Header):
    """ 
    Encapsulates methods specific to the astropy.io.fits.Header class.
    Note : for all methods, the axes are always given in their numpy array format, not in the fits header format. For
    example, axis=0 targets the first numpy array axis, and therefore the last header axis (e.g. 3 for a cube). Values
    of 0, 1 and 2 target respectively z, y and x.
    """
    
    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: Header) -> bool:
        keys_equal = list(self.keys()) == list(other.keys())

        for key, value in self.items():
            if value != other[key] and key != "COMMENT":
                values_equal = False
                break
        else:
            values_equal = True

        return keys_equal and values_equal

    @property
    def wcs_object(self) -> WCS:
        """
        Converts the Header to a WCS object. This is useful for plotting as the returned object can simply be passed to
        the projection argument of the matplotlib/graphinglib figure.

        Returns
        -------
        WCS
            WCS object corresponding to the Header, that may be passed as the `projection` argument of a figure.
        """
        return WCS(self)

    def _h_axis(self, axis: int) -> int:
        """
        Converts a numpy axis to a header axis.

        Parameters
        ----------
        axis : int
            Axis to convert to a header axis.

        Returns
        -------
        int
            Axis converted to a header axis.
        """
        h_axis = self["NAXIS"] - axis
        return h_axis

    def bin(self, bins: list[int] | tuple[int, int] | tuple[int, int, int]) -> Self:
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
        Self
            Binned Header.
        """
        assert list(bins) == list(filter(lambda val: val >= 1 and isinstance(val, int), bins)), \
            f"{C.RED}All values in bins must be greater than or equal to 1 and must be integers.{C.OFF}"
        
        header_copy = self.copy()
        for ax, bin_ in enumerate(bins):
            h_ax = self._h_axis(ax)
            if f"CDELT{h_ax}" in list(self.keys()):
                header_copy[f"CDELT{h_ax}"] *= bin_
            if f"CRPIX{h_ax}" in list(self.keys()):
                header_copy[f"CRPIX{h_ax}"] = (self[f"CRPIX{h_ax}"] - 0.5) / bin_ + 0.5
            if f"NAXIS{h_ax}" in list(self.keys()):
                header_copy[f"NAXIS{h_ax}"] = self[f"NAXIS{h_ax}"] // bin_
        
        return header_copy

    def flatten(self, axis: int) -> Self:
        """
        Flattens a Header by removing an axis. The remaining axes are placed so they stay coherent (start at 1 and
        increment by constant steps of 1). This method is safer than the _remove_axis method.

        Parameters
        ----------
        axis : int
            Axis to flatten.

        Returns
        -------
        Self
            Flattened header with the remaining data.
        """
        new_header = self.copy()
        for i in range(axis):
            # Swap axes to place the axis to remove at indice 0
            new_header = new_header.swap_axes(axis - i - 1, axis - i)

        # Erase the axis
        new_header = new_header._remove_axis(0)

        return new_header

    def _remove_axis(self, axis: int) -> Self:
        """
        Removes an axis from a Header. The remaining axes are not moved so incoherent headers, without certain axes, may
        occur (e.g. AXIS1 and AXIS3, but no AXIS2). The flatten method is safer to use than this one.

        Parameters
        ----------
        axis : int
            Axis to remove.

        Returns
        -------
        Self
            Header with the removed axis.
        """
        new_header = self.copy()
        h_axis = str(self._h_axis(axis))
        for key in deepcopy(list(new_header.keys())):
            if key and key[-1] == h_axis:
                new_header.pop(key)
        
        new_header["NAXIS"] -= 1

        return new_header

    @FitsFile.silence_function
    def swap_axes(self, axis_1: int, axis_2: int) -> Self:
        """
        Switches a Header's axes to fit a FitsFile object with swapped axes.
        
        Parameters
        ----------
        axis_1 : int
            Source axis.
        axis_2 : int
            Destination axis.
        
        Returns
        -------
        Self
            Header with the switched axes.
        """
        # Make header readable keywords
        h_axis_1, h_axis_2 = self._h_axis(axis_1), self._h_axis(axis_2)
        new_header = self.copy()

        for key in deepcopy(list(self.keys())):
            if key:
                if key[-1] == str(h_axis_1):
                    new_header[f"{key[:-1]}{h_axis_2}-"] = new_header.pop(key)
                elif key[-1] == str(h_axis_2):
                    new_header[f"{key[:-1]}{h_axis_1}-"] = new_header.pop(key)
        
        # The modified header keywords are temporarily named with the suffix "-" to prevent duplicates during the
        # process
        # After the process is done, the suffix is removed
        for key in deepcopy(list(new_header.keys())):
            if key and key[-1] == "-":
                new_header[key[:-1]] = new_header.pop(key)

        return new_header

    def invert_axis(self, axis: int) -> Self:
        """
        Inverts a Header along an axis.

        Parameters
        ----------
        axis : int
            Axis along which the info needs to be inverted.

        Returns
        -------
        Self
            Header with the inverted axis.
        """
        new_header = self.copy()
        h_axis = self._h_axis(axis)
        new_header[f"CDELT{h_axis}"] *= -1
        new_header[f"CRPIX{h_axis}"] = self[f"NAXIS{h_axis}"] - self[f"CRPIX{h_axis}"] + 1
        return new_header
    
    def crop_axes(self, slices: tuple[slice | int]) -> Self:
        """
        Crops the Header to account for a cropped FitsFile.

        Parameters
        ----------
        slices : tuple[slice | int]
            Slices to crop each axis. An integer slice will not crop the axis.
        
        Returns
        -------
        Self
            Cropped Header.
        """
        new_header = self.copy()
        for i, s in enumerate(slices):
            if isinstance(s, slice):
                h_axis = self._h_axis(i)
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else self[f"NAXIS{h_axis}"]
                new_header[f"CRPIX{h_axis}"] -= start
                new_header[f"NAXIS{h_axis}"] = stop - start

        return new_header
    
    def concatenate(self, other: Header, axis: int) -> Self:
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
        Self
            Concatenated Header.
        """
        new_header = self.copy()
        h_axis = self._h_axis(axis)
        new_header[f"NAXIS{h_axis}"] += other[f"NAXIS{h_axis}"]
        return new_header

    def get_coordinate(self, value: float, axis: int=0) -> int:
        """
        Gives the coordinate closest to the specified value, along the given axis. Currently supported projections are
        the Global Sinusoidal projection (GLS) and the cartesian projection (CAR).
        
        Parameters
        ----------
        value : float
            Value to determine the coordinate. This can be a value in the range of any axis.
        axis : int, default=0
            Axis along which to get the coordinate. The default axis (0) gives the coordinate along a cube's spectral
            axis.

        Returns
        -------
        int
            Coordinate closest to the specified value.
        """
        h_axis = self._h_axis(axis)
        if self[f"CTYPE{h_axis}"] == "RA---GLS":
            DEC_axis = list(self.keys())[list(self.values()).index("DEC--GLS")][5:]
            frame_number = (value - self[f"CRVAL{h_axis}"]) \
                         / (self[f"CDELT{h_axis}"]/cos(radians(self[f"CRVAL{DEC_axis}"]))) \
                         + self[f"CRPIX{h_axis}"]
        elif self[f"CTYPE{h_axis}"][-3:] in ["CAR", "LSR", "    "]:
            frame_number = (value - self[f"CRVAL{h_axis}"]) / self[f"CDELT{h_axis}"] + self[f"CRPIX{h_axis}"]
        else:
            raise NotImplementedError(C.RED + f"CTYPE {self[f"CTYPE{h_axis}"]} not supported." + C.OFF)
        rounded_frame = round(frame_number)
        return rounded_frame

    def get_value(self, coordinate: int, axis: int=0) -> float:
        """
        Gives the value associated with the specified coordinate, along the given axis. Currently supported projections
        are the Global Sinusoidal projection (GLS) and the cartesian projection (CAR).
        
        Parameters
        ----------
        coordinate : int
            Coordinate to determine the value. This should be a coordinate in the range of any axis.
        axis : int, default=0
            Axis along which to get the value at the specified coordinate. For example, the default axis (0) gives the
            value along a cube's spectral axis, if the header is associated with a Cube.

        Returns
        -------
        float
            Value at the given coordinate.
        """
        h_axis = self._h_axis(axis)
        if self[f"CTYPE{h_axis}"] == "RA---GLS":
            DEC_axis = list(self.keys())[list(self.values()).index("DEC--GLS")][5:]
            value = (coordinate - self[f"CRPIX{h_axis}"]) \
                  * (self[f"CDELT{h_axis}"]/cos(radians(self[f"CRVAL{DEC_axis}"]))) \
                  + self[f"CRVAL{h_axis}"]
        elif self[f"CTYPE{h_axis}"][-3:] in ["CAR", "LSR", "    "]:
            value = (coordinate - self[f"CRPIX{h_axis}"]) * self[f"CDELT{h_axis}"] + self[f"CRVAL{h_axis}"]
        else:
            raise NotImplementedError(C.RED + f"CTYPE {self[f"CTYPE{h_axis}"]} not supported." + C.OFF)
        return value
