import numpy as np
import pyregion
from astropy.io.fits.header import Header


class Mask:
    """
    Encapsulate the methods specific to masks.
    The returned numpy arrays are returned as integers to allow operations between masks to combine them. Operators such
    as & (bitwise AND), | (bitwise OR) or ^ (bitwise XOR) may be of use. The returned mask may then be multiplied with
    the corresponding data.
    """

    def __init__(self, image_shape: tuple[int, int]):
        """
        Initializes a Mask object.

        Parameters
        ----------
        image_shape : tuple[int, int]
            Shape of the image for which the mask will be used.
        """
        self.image_shape = image_shape
    
    def _get_numpy_mask(self, region: pyregion.core.ShapeList) -> np.ndarray:
        """
        Gives the numpy mask of the current Mask object.
        
        Parameters
        ----------
        region : pyregion.core.ShapeList
            Region with which the numpy mask will be made.

        Returns
        -------
        mask : np.ndarray
            Exported mask.
        """
        mask = region.get_mask(shape=self.image_shape)
        return mask
    
    def open_as_image_coord(self, filename: str, header: Header):
        """
        Opens a FITS file as an image coordinate mask.

        Parameters
        ----------
        filename : str
            Name of the FITS file.
        header : Header
            Header of the FITS file.

        Returns
        -------
        mask : np.ndarray
            Exported mask.
        """
        region = pyregion.open(filename).as_imagecoord(header)
        return self._get_numpy_mask(region)

    def circle(self, center: tuple[float, float], radius: float) -> np.ndarray:
        """
        Creates a circular mask.

        Parameters
        ----------
        center : tuple[float, float]
            Center of the circular mask.
        radius : float
            Radius of the circular mask.

        Returns
        -------
        mask : np.ndarray
            Generated mask.
        """
        region_id = f"image;circle({center[0]},{center[1]},{radius})"
        region = pyregion.parse(region_id)
        return self._get_numpy_mask(region)
    
    def ellipse(
            self,
            center: tuple[float, float],
            semi_major_axis: float,
            semi_minor_axis: float,
            angle: float=0
    ) -> np.ndarray:
        """
        Creates an elliptical mask.

        Parameters
        ----------
        center : tuple[float, float]
            Center of the circular mask.
        semi_major_axis : float
            Length in pixels of the semi-major axis.
        semi_minor_axis : float
            Length in pixels of the semi-minor axis.
        angle : float, default=0
            Angle of the shape, in degrees, relative to the position where the semi-major axis is parallel to the x
            axis. Increasing values rotates the shape clockwise.

        Returns
        -------
        mask : np.ndarray
            Generated mask.
        """
        region_id = f"image;ellipse({center[0]},{center[1]},{semi_major_axis},{semi_minor_axis},{angle})"
        region = pyregion.parse(region_id)
        return self._get_numpy_mask(region)

    def rectangle(self, center: tuple[float, float], length: float, height: float, angle: float=0) -> np.ndarray:
        """
        Creates a rectangular mask.

        Arguments
        ---------
        center : tuple[float, float]
            Center of the circular mask.
        radius : float
            Radius of the circular mask.
        angle : float, default=0
            Angle of the shape, in degrees, relative to the position where the length axis is parallel to the x axis.
            Increasing values rotates the shape clockwise.

        Returns
        -------
        mask : np.ndarray
            Generated mask.
        """
        region_id = f"image;box({center[0]},{center[1]},{length},{height},{angle})"
        region = pyregion.parse(region_id)
        return self._get_numpy_mask(region)

    def polygon(self, vertices: list[tuple[float, float]]) -> np.ndarray:
        """
        Creates a polygon mask.

        Arguments
        ---------
        vertices : list[tuple[float, float]]
            Vertices of the polygon. Each element is a vertex and is defined by its (x,y) coordinates.

        Returns
        -------
        mask : np.ndarray
            Generated mask.
        """
        region_id = f"image;polygon{sum(vertices, ())}"
        region = pyregion.parse(region_id)
        return self._get_numpy_mask(region)