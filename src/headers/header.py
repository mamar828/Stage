from __future__ import annotations
from astropy.io import fits
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
        Bins a header.

        Parameters
        ----------
        bins : tuple[int, int] | tuple[int, int, int]
            Number of pixels to be binned together along each axis (1-3). The size of the tuple varies depending on the
            fits file's number of dimensions. A value of 1 results in the axis not being binned.

        Returns
        -------
        header : Header
            Binned header.
        """
        assert list(bins) == list(filter(lambda val: val >= 1 and isinstance(val, int), bins)), \
            f"{C.RED+C.BOLD}All values in bins must be greater than or equal to 1 and must be integers.{C.END}"
        
        header_copy = self.copy()
        for ax, bin_ in zip(range(1, len(bins) + 1), bins):
            if f"CDELT{ax}" in list(self.keys()):
                header_copy[f"CDELT{ax}"] *= bin_
            if f"CRPIX{ax}" in list(self.keys()):
                header_copy[f"CRPIX{ax}"] = (self[f"CRPIX{ax}"] - 0.5) / bin_ + 0.5
        
        return header_copy
