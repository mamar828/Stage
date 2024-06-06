class DS9Coords:
    """
    Encapsulates the methods specific to SAOImage ds9 coordinates and their conversion.
    To properly use this class, an object may be created, then unpacked to be given to methods that require normal ints.
    Example : spectrum = Cube[:,*DS9Coords(5,10)]
    will correctly slice the Cube at the specified coordinates, which returns a Spectrum.
    """

    def __init__(self, *coordinates: tuple[int]):
        """
        Initialize a DS9Coords object.

        Parameters
        ----------
        coordinates : tuple[int]
            Coordinates to initialize the object with. These are given in the same order as in DS9.
        """
        self.coords = list(coordinates)
    
    def __len__(self):
        return len(self.coords)

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
            coord = self.coords[index] - 1
            return coord
        else:
            # Allows the use of the unpacking operator
            raise IndexError
