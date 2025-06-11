from __future__ import annotations


class DS9Coords:
    """
    Encapsulates the methods specific to SAOImage ds9 coordinates and their conversion.
    To properly use this class, an object may be created, then unpacked to be given to methods that require normal ints.
    This allows for any function to use this class as the unpacked values are ints.

    Examples
    --------
    spectrum = Cube[:,*DS9Coords(5,10)]
    will correctly slice the Cube at the specified coordinates, which returns a Spectrum.

    The following lines show how useful this class is for precisely slicing a FitsFile object. To get the value at the
    (x, y, z) = (8, 17, 225) on SAOImage ds9, the following code is equivalent :
    using DS9Coords :   print(Cube[*DS9Coords(8, 17, 225)])
    without DS9Coords : print(Cube[224, 16, 7])
    In the former case, the coordinates can intuitively be entered and in the latter, an awkward double conversion is
    required.
    """

    def __init__(self, *coordinates: tuple[int]):
        """
        Initialize a DS9Coords object.

        Parameters
        ----------
        coordinates : tuple[int]
            Coordinates to initialize the object with. These are given in the same order as in DS9.
        """
        self.data = list(coordinates)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key: int) -> int:
        """
        Gives the value of the specified key by converting the DS9 coordinate to a numpy one. The conversion is made by
        inverting the coordinates order (e.g. x,y -> y,x) as numpy indexing starts with the "last index", then by
        removing 1 because DS9 indexing starts at (1,1), and not (0,0).
        """
        # Roll axis to account for the coordinate switch
        index = len(self) - 1 - key
        if index >= 0:
            # Reduce axis to account for indexing differences
            coord = self.data[index] - 1
            return coord
        else:
            # Allows the use of the unpacking operator
            raise IndexError

    def __str__(self):
        return f"DS9Coords({', '.join(map(str, self.data))})"

    @classmethod
    def from_python(cls, *coordinates: tuple[int]) -> DS9Coords:
        """
        Create a DS9Coords object from Python coordinates.

        Parameters
        ----------
        coordinates : tuple[int]
            Coordinates to initialize the object with. These are given in the same order as in Python.

        Returns
        -------
        DS9Coords
            A new DS9Coords object initialized with the provided coordinates.
        """
        coords = coordinates[::-1]
        coords = [c + 1 for c in coords]
        return cls(*coords)
