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
            if f"CDELT{ax}" in list(self.keys()):
                header_copy[f"CDELT{ax}"] *= bin_
            if f"CRPIX{ax}" in list(self.keys()):
                header_copy[f"CRPIX{ax}"] = (self[f"CRPIX{ax}"] - 0.5) / bin_ + 0.5
        
        return header_copy

    def get_flattened(self) -> Header:
        """
        Get the adaptation of a Cube Header for a Map Header by removing the spectral axis.

        Returns
        -------
        header : Header
            Header with the same data but with the third axis removed.
        """
        header = self.header.copy()
        wcs = WCS(header)
        wcs.sip = None
        wcs = wcs.dropaxis(2)
        header = wcs.to_header(relax=True)
        return header

    def get_switched(self, axis_1: int, axis_2: int) -> Header:
        """
        Gets a Header with switched axes to fit a Cube whose axes were also swapped.
        
        Arguments
        ---------
        axis_1: int
            Source axis.
        axis_2: int
            Destination axis.
        
        Returns
        -------
        header : Header
            Header with the switched axes.
        """
        h_axis_1, h_axis_2 = axis_1 + 1, axis_2 + 1             # The header uses 1-based indexing

        new_header = self.copy()

        for header_element in deepcopy(list(self.keys())):
            if header_element[-1] == str(h_axis_1):
                new_header[f"{header_element[:-1]}{h_axis_2}-"] = new_header.pop(header_element)
            elif header_element[-1] == str(h_axis_2):
                new_header[f"{header_element[:-1]}{h_axis_1}-"] = new_header.pop(header_element)
        
        # The modified header keywords are temporarily named with the suffix "-" to prevent duplicates during the
        # process
        # After the process is done, the suffix is removed
        for header_element in deepcopy(list(new_header.keys())):
            if header_element[-1] == "-":
                new_header[header_element[:-1]] = new_header.pop(header_element)
        return new_header

    def get_inverted(self, axis: int) -> Header:
        """
        Get a Header inverted along an axis.

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
        h_axis = 3 - axis
        new_header[f"CDELT{h_axis}"] *= -1
        new_header[f"CRPIX{h_axis}"] = self.data.shape[axis] - self.header[f"CRPIX{h_axis}"] + 1
        return new_header
