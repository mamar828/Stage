class Coord:
    """
    This class defines a Coord object used for working with coordinates and their different representations. This class
    is not meant to be instantiated directly, but rather serves as a base class for RA and DEC.
    """
    def __init__(self, degrees: float):
        """
        Initializes a Coord object.

        Parameters
        ----------
        degrees : float
            Value in degrees associated with the Coord object.
        """
        self.degrees = degrees

    def __str__(self) -> str:
        return f"{self.__class__.__name__} : {self.degrees:.4f}°, {self.sexagesimal}"


class RA(Coord):
    """
    This class implements a Coord for right ascension.
    """
    @classmethod
    def from_sexagesimal(cls, value: str):
        """
        Creates a RA object from a sexagesimal string.

        Parameters
        ----------
        value : str
            Sexagesimal string in the format "hours:minutes:seconds".

        Returns
        -------
        RA
            New RA object representing the given sexagesimal value.
        """
        hours, minutes, seconds = [float(val) for val in value.split(":")]
        degrees = (hours*3600 + minutes*60 + seconds) / (24*3600) * 360
        return cls(degrees)

    @property
    def sexagesimal(self) -> str:
        """
        Returns the sexagesimal representation of the RA object.

        Returns
        -------
        str
            Sexagesimal representation of the RA object in the format "hours:minutes:seconds".
        """
        total_seconds = self.degrees / 360 * (24*3600)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) / 3600 * 60)
        seconds = total_seconds - hours * 3600 - minutes * 60
        return f"{hours}:{minutes:02d}:{seconds:06.3f}"


class DEC(Coord):
    """
    This class implements a Coord for declination.
    """
    @classmethod
    def from_sexagesimal(cls, value: str):
        """
        Creates a DEC object from a sexagesimal string.

        Parameters
        ----------
        value : str
            Sexagesimal string in the format "degrees:minutes:seconds".

        Returns
        -------
        DEC
            New DEC object representing the given sexagesimal value.
        """
        whole_degrees, minutes, seconds = [float(val) for val in value.split(":")]
        degrees = whole_degrees + minutes / 60 + seconds / 3600
        return cls(degrees)

    @property
    def sexagesimal(self) -> str:
        """
        Returns the sexagesimal representation of the DEC object.

        Returns
        -------
        str
            Sexagesimal representation of the DEC object in the format "degrees:minutes:seconds".
        """
        whole_degrees = int(self.degrees)
        minutes = int((self.degrees % 1) * 60)
        seconds = ((self.degrees % 1) * 60 - minutes) * 60
        return f"{whole_degrees}:{minutes:02d}:{seconds:06.3f}"
