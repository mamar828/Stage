from src.hdu.fits_file import FitsFile
from src.hdu.array_3d import Array3D
from src.headers.header import Header


class Cube(FitsFile):
    """
    Encapsulates the methods specific to data cubes.
    """

    def __init__(self, value: Array3D, header: Header=None):
        """
        Initialize a Cube object.

        Parameters
        ----------
        value : Array3D
            The values of the Cube.
        header : Header, default=None
            The header of the Cube.
        """
        self.value = value
        self.header = header
