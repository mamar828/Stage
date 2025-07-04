from __future__ import annotations
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from typing import Self
from colorist import BrightColor as C


class Header(fits.Header):
    """
    Encapsulates methods specific to the astropy.io.fits.Header class.

    .. warning::
        For all methods, the axes are always given in their numpy array format, not in the fits header format. For
        example, axis=0 targets the first numpy array axis, and therefore the last header axis (e.g. 3 for a cube).
        Values of 0, 1 and 2 target respectively the z, y and x axes.
    """

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: Header) -> bool:
        keys_equal = list(self.keys()) == list(other.keys())

        for key, value in self.items():
            if value != other[key] and key not in ["", "COMMENT"]:
                values_equal = False
                break
        else:
            values_equal = True

        return keys_equal and values_equal

    @property
    def wcs(self) -> WCS:
        """
        Returns the WCS object of the Header. This is useful for plotting as the returned object can simply be passed to
        the projection argument of the matplotlib/graphinglib figure.
        """
        return WCS(self)

    @property
    def celestial(self) -> Self:
        """
        Returns a Header containing only the celestial axes (RA, DEC, etc.) of the WCS.
        """
        celestial_wcs = self.wcs.celestial
        return self._update_wcs(celestial_wcs)

    @property
    def spectral(self) -> Self:
        """
        Returns a Header containing only the spectral axis of the WCS.
        """
        spectral_wcs = self.wcs.spectral
        return self._update_wcs(spectral_wcs)

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

    def _update_wcs(self, wcs: WCS) -> Self:
        """
        Updates the Header with a WCS object. This method replaces only the WCS-related keywords in the Header with
        those from the WCS object, while preserving any other metadata that may be present.
        """
        old_wcs_header = self.wcs.to_header()       # this contains all old WCS keywords
        new_wcs_header = wcs.to_header()            # this contains the new WCS keywords to update
        new_wcs_header_keys = list(new_wcs_header.keys()) # this gives all the new WCS keywords
        new_header = self.copy()

        # Look at each old WCS key and update it with the new value if it exists in the new WCS header
        # If it doesn't exist in the new WCS header, remove it from the new header
        # This ensures that the new header only contains relevant WCS keywords
        for old_key in old_wcs_header.keys():
            if old_key in new_wcs_header_keys:
                # Update the key with the new value and preserve order and comments
                new_header.set(old_key, new_wcs_header[old_key], new_wcs_header.comments[old_key])
            else:
                # Remove the old key if it doesn't exist in the new WCS
                new_header.remove(old_key, ignore_missing=True)

        for h_axis in range(1, new_header["NAXIS"] + 1):
            if h_axis > wcs.naxis:
                new_header.remove(f"NAXIS{h_axis}", ignore_missing=True)
                new_header.remove(f"CROTA{h_axis}", ignore_missing=True)
        new_header["NAXIS"] = wcs.naxis  # Update the NAXIS keyword to the new WCS naxis
        return new_header

    def bin(self, bins: int | tuple[int, int] | tuple[int, int, int]) -> Self:
        """
        Bins a Header.

        Parameters
        ----------
        bins : int | tuple[int, int] | tuple[int, int, int]
            Number of pixels to be binned together along each axis (1-3). The size of the tuple varies depending on the
            fits file's number of dimensions. A value of 1 results in the axis not being binned. Read the note in the
            declaration of this function to properly indicate the axes.

        Returns
        -------
        Self
            Binned Header.
        """
        list_bins = list([bins] if not isinstance(bins, tuple) else bins)
        if len(list_bins) != self["NAXIS"]:
            raise ValueError(
                f"{C.RED}The number of bins must match the number of axes in the Header. "
                f"Expected {self['NAXIS']} bins, got {len(list_bins)}.{C.OFF}"
            )
        if any(bin_i < 1 or not isinstance(bin_i, int) for bin_i in list_bins):
            raise ValueError(
                f"{C.RED}All values in bins must be greater than or equal to 1 and must be integers.{C.OFF}"
            )
        binned_wcs = self.wcs.slice([slice(None, None, bin_i) for bin_i in list_bins])
        new_header = self._update_wcs(binned_wcs)
        new_naxes = {f"NAXIS{self._h_axis(i)}": self[f"NAXIS{self._h_axis(i)}"] // bin_i
                     for i, bin_i in enumerate(list_bins)}
        new_header.update(new_naxes)
        return new_header

    def swap_axes(self, axis_1: int, axis_2: int) -> Self:
        """
        Switches a Header's axes to fit a FitsFile object with swapped axes. The Header is modified in place.

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
        swapped_wcs = self.wcs.swapaxes(self._h_axis(axis_1)-1, self._h_axis(axis_2)-1) # this uses 0-based indexing
        new_header = self._update_wcs(swapped_wcs)
        # Update NAXIS keywords
        new_header[f"NAXIS{self._h_axis(axis_1)}"] = self[f"NAXIS{self._h_axis(axis_2)}"]
        new_header[f"NAXIS{self._h_axis(axis_2)}"] = self[f"NAXIS{self._h_axis(axis_1)}"]
        return new_header

    def invert_axis(self, axis: int) -> Self:
        """
        Inverts one axis of the Header.

        Parameters
        ----------
        axis : int
            Axis to invert. The Header is modified in place.

        Returns
        -------
        Self
            Header with the inverted axis.
        """
        wcs = self.wcs.deepcopy()
        axis_index = self._h_axis(axis) - 1  # convert to 0-based indexing
        naxis = self[f"NAXIS{self._h_axis(axis)}"]

        if wcs.wcs.has_pc():
            wcs.wcs.pc[:, axis_index] *= -1
        else:
            wcs.wcs.cd[:, axis_index] *= -1
        wcs.wcs.crpix[axis_index] = naxis + 1 - wcs.wcs.crpix[axis_index]

        return self._update_wcs(wcs)

    def slice(self, slices: tuple[slice]) -> Self:
        """
        Slices the Header according to the given slices.

        Parameters
        ----------
        slices : tuple[slice]
            Slices to crop each axis, as for numpy arrays.

            .. warning::
                Integer slices are not supported. If you want to crop an axis, use the :attr:`celestial` or
                :attr:`spectral` properties to get a Header with only the celestial or spectral axes.

        Returns
        -------
        Self
            Sliced Header.
        """
        if not all(isinstance(s, slice) for s in slices):
            raise ValueError(f"{C.RED}All slices must be of type 'slice'.{C.OFF}")

        sliced_wcs = self.wcs.slice(slices)
        new_header = self._update_wcs(sliced_wcs)
        # Update NAXIS keywords
        for i, s in enumerate(slices):
            h_axis = self._h_axis(i)
            start = s.start if s.start is not None else 0
            stop = s.stop if s.stop is not None else self[f"NAXIS{h_axis}"]
            new_header[f"NAXIS{h_axis}"] = stop - start
        return new_header

    def concatenate(self, other: Header, axis: int) -> Self:
        """
        Concatenates two headers along an axis. The Header closest to the origin should be the one to call this method.
        This method is used if a FitsFile whose header was previously cropped (with the :meth:`slice` method) needs to
        be re-concatenated. The FitsFiles are considered directly next to each other.

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

    def pixel_to_world(self, pixel_coords: list[float] | np.ndarray[float]) -> np.ndarray:
        """
        Converts pixel coordinates to world coordinates using the WCS of the Header. This method wraps the
        `astropy.wcs.WCS.pixel_to_world` method, but the order of the `x` and `y` coordinates is inverted.

        Parameters
        ----------
        pixel_coords : list[float] | np.ndarray[float]
            Pixel coordinates to convert to world coordinates. This should be a 2D array with shape (n, m) where n is
            the number of coordinates and the second dimension contains the m-dimensional coordinates in numpy format.

            .. warning::
                The input pixel coordinates should be in the format (y, x) as per numpy's convention, not (x, y) as per
                the FITS standard. This means that the first column should contain the y pixel coordinates and the
                second column should contain the x pixel coordinates. For using ds9 coordinates, see the
                `src.coordinates.ds9_coords.DS9Coords` class.

        Returns
        -------
        np.ndarray
            World coordinates corresponding to the input pixel coordinates. The output will be a 2D array with shape
            (n, m) where the second dimension contains the world coordinates (e.g. [RA, DEC, velocity] for an equatorial
            conversion).
        """
        if isinstance(pixel_coords, list):
            pixel_coords = np.array(pixel_coords)
        if pixel_coords.ndim == 1:
            pixel_coords = pixel_coords.reshape(1, -1)
        elif pixel_coords.ndim != 2:
            raise ValueError("pixel_coords must be a 2D array.")

        transposed_coords = pixel_coords.T[::-1]
        world_coords = np.array(self.wcs.pixel_to_world_values(*transposed_coords)).T
        return world_coords

    def world_to_pixel(self, world_coords: list[float] | np.ndarray[float]) -> np.ndarray:
        """
        Converts world coordinates to pixel coordinates using the WCS of the Header. This method wraps the
        `astropy.wcs.WCS.world_to_pixel` method.

        Parameters
        ----------
        world_coords : list[float] | np.ndarray[float]
            World coordinates to convert to pixel coordinates. This should be a 2D array with shape (n, m) where n is
            the number of coordinates and the second dimension contains the m-dimensional coordinates in standard
            celestial format (e.g. [RA, DEC, velocity] for an equatorial conversion).

        Returns
        -------
        np.ndarray
            Pixel coordinates corresponding to the input world coordinates. The output will be a 2D array with shape
            (n, m) where the second dimension contains the pixel coordinates (e.g. [y, x] for a 2D image).

            .. warning::
                The output pixel coordinates will be in the format (y, x) as per numpy's convention, not (x, y) as per
                the FITS standard. This means that the first column will contain the y pixel coordinates and the
                second column will contain the x pixel coordinates. For converting to ds9 coordinates, see the
                `src.coordinates.ds9_coords.DS9Coords` class.
        """
        if isinstance(world_coords, list):
            world_coords = np.array(world_coords)
        if world_coords.ndim == 1:
            world_coords = world_coords.reshape(1, -1)
        elif world_coords.ndim != 2:
            raise ValueError("world_coords must be a 2D array.")

        pixel_coords = self.wcs.world_to_pixel_values(*world_coords.T)
        pixel_coords = np.array(pixel_coords)[::-1].T
        return pixel_coords
