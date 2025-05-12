import numpy as np
import pyregion
from astropy.io.fits.header import Header


class Mask:
    """
    Encapsulate the methods specific to masks.
    The returned numpy arrays are returned as integers to allow operations between masks to combine them. Operators such
    as & (bitwise AND), | (bitwise OR) or ^ (bitwise XOR) may be of use. The returned mask may then be multiplied with
    the corresponding data.
    Note that all values must be given in pixels.
    """

    def __init__(self, image_shape: tuple[int, int]):
        """
        Initializes a Mask object.

        Parameters
        ----------
        image_shape : tuple[int, int]
            Shape (x, y) of the image for which the mask will be used.
        """
        self.image_shape = image_shape
    
    def open_as_image_coord(self, filename: str, header: Header) -> np.ndarray:
        """
        Opens a .reg file as an image coordinate mask.

        Parameters
        ----------
        filename : str
            Name of the .reg file containing the mask.
        header : Header
            Header of the FITS file with which the region was created. This is used for converting from WCS to image
            coordinates. Make sure that the Mask was initialized with the same shape as the FITS file.

        Returns
        -------
        np.ndarray
            Exported mask represented by the .reg file.
        """
        region = pyregion.open(filename).as_imagecoord(header)
        return self._get_numpy_mask(region)

    def circle(self, center: tuple[float, float], radius: float) -> np.ndarray:
        """
        Creates a circular mask.

        Parameters
        ----------
        center : tuple[float, float]
            Center (x, y) of the circular mask.
        radius : float
            Radius of the circular mask.

        Returns
        -------
        np.ndarray
            Generated circular mask.
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
            Center (x, y) of the elliptical mask.
        semi_major_axis : float
            Length in pixels of the semi-major axis. With an angle of zero, the semi-major axis is parallel to the x
            axis.
        semi_minor_axis : float
            Length in pixels of the semi-minor axis. With an angle of zero, the semi-minor axis is parallel to the y
            axis.
        angle : float, default=0
            Angle of the shape, in degrees, relative to the position where the semi-major axis is parallel to the x
            axis. Increasing values rotates the shape clockwise.

        Returns
        -------
        np.ndarray
            Generated elliptical mask.
        """
        region_id = f"image;ellipse({center[0]},{center[1]},{semi_major_axis},{semi_minor_axis},{angle})"
        region = pyregion.parse(region_id)
        return self._get_numpy_mask(region)

    def rectangle(self, center: tuple[float, float], length: float, height: float, angle: float=0) -> np.ndarray:
        """
        Creates a rectangular mask.

        Parameters
        ----------
        center : tuple[float, float]
            Center (x, y) of the rectangular mask.
        length : float
            Length of the rectangular mask.  With an angle of zero, the length is parallel to the x axis.
        height : float
            Height of the rectangular mask. With an angle of zero, the height is parallel to the y axis.
        angle : float, default=0
            Angle of the shape, in degrees, relative to the position where the length axis is parallel to the x axis.
            Increasing values rotates the shape clockwise.

        Returns
        -------
        np.ndarray
            Generated rectangular mask.
        """
        region_id = f"image;box({center[0]},{center[1]},{length},{height},{angle})"
        region = pyregion.parse(region_id)
        return self._get_numpy_mask(region)

    def polygon(self, vertices: list[tuple[float, float]]) -> np.ndarray:
        """
        Creates a polygon mask.

        Parameters
        ----------
        vertices : list[tuple[float, float]]
            Vertices of the polygon. Each element is a vertex and is defined by its (x, y) coordinates. The generated
            polygon links the given vertices in the same order as given in the list and links the last vertice with the
            first.

        Returns
        -------
        np.ndarray
            Generated polygonal mask.
        """
        region_id = f"image;polygon{sum(vertices, ())}"
        region = pyregion.parse(region_id)
        return self._get_numpy_mask(region)
    
    def ring(self, center: tuple[float, float], inner_radius: float, outer_radius: float) -> np.ndarray:
        """
        Creates a ring mask. The outputted ring has a width of (outer_radius - inner_radius).

        Parameters
        ----------
        center : tuple[float, float]
            Center (x, y) of the ring.
        inner_radius : float
            Inner radius of the ring.
        outer_radius : float
            Outer radius of the ring.

        Returns
        -------
        np.ndarray
            Generated ring mask.
        """
        inner_circle = self.circle(center, inner_radius)
        outer_circle = self.circle(center, outer_radius)
        return outer_circle ^ inner_circle

    def _get_numpy_mask(self, region: pyregion.core.ShapeList) -> np.ndarray:
        """
        Gives the numpy mask of the provided region.
        
        Parameters
        ----------
        region : pyregion.core.ShapeList
            Region with which the numpy mask will be made.

        Returns
        -------
        np.ndarray
            Exported mask.
        """
        mask = region.get_mask(shape=self.image_shape)
        return mask
