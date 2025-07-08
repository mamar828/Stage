from __future__ import annotations
from astropy.coordinates import SkyCoord


def equatorial_to_galactic(ra: RA, dec: DEC) -> tuple[l, b]:
    """
    Converts equatorial coordinates (RA, DEC) to galactic coordinates (l, b).

    Parameters
    ----------
    ra : RA
        Right Ascension coordinate.
    dec : DEC
        Declination coordinate.

    Returns
    -------
    tuple[l, b]
        Galactic coordinates (l, b).
    """
    equatorial_coords = SkyCoord(ra=ra.degrees, dec=dec.degrees, unit="deg", frame="icrs")
    galactic_coords = equatorial_coords.galactic
    return l(galactic_coords.l.deg), b(galactic_coords.b.deg)
    # Placeholder for actual conversion logic
    # This should be replaced with the actual conversion algorithm
    return l(0.0), b(0.0)

def galactic_to_equatorial(l: l, b: b) -> tuple[RA, DEC]:
    """
    Converts galactic coordinates (l, b) to equatorial coordinates (RA, DEC).

    Parameters
    ----------
    l : l
        Galactic longitude coordinate.
    b : b
        Galactic latitude coordinate.

    Returns
    -------
    tuple[RA, DEC]
        Equatorial coordinates (RA, DEC).
    """
    galactic_coords = SkyCoord(l=l.degrees, b=b.degrees, unit="deg", frame="galactic")
    equatorial_coords = galactic_coords.icrs
    return RA(equatorial_coords.ra.deg), DEC(equatorial_coords.dec.deg)


class Coord:
    """
    This class defines a `Coord` object used for working with coordinates and their different representations. This
    class is not meant to be instantiated directly, but rather serves as a base class for specific celestial
    coordinates.
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

    @classmethod
    def from_sexagesimal(cls, value: str) -> Coord:
        """
        Creates a Coord object from a string in the format "degrees:minutes:seconds" to degrees. This method is invalid
        for the RA class, which uses hours instead of degrees..

        Parameters
        ----------
        value : str
            Sexagesimal string in the format "degrees:minutes:seconds".

        Returns
        -------
        float
            New Coord object representing the given sexagesimal string.
        """
        whole_degrees, minutes, seconds = [float(val) for val in value.split(":")]
        degrees = whole_degrees + minutes / 60 + seconds / 3600
        return cls(degrees)

    @property
    def sexagesimal(self) -> str:
        """
        Returns the sexagesimal representation of the Coord object.

        Returns
        -------
        str
            Sexagesimal representation of the Coord object in the format "degrees:minutes:seconds".
        """
        whole_degrees = int(self.degrees)
        minutes = int((self.degrees % 1) * 60)
        seconds = ((self.degrees % 1) * 60 - minutes) * 60
        return f"{whole_degrees}:{minutes:02d}:{seconds:06.3f}"


class RA(Coord):
    """
    This class implements a `Coord` for right ascension.
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
    This class implements a `Coord` for declination.
    """
    pass


class l(Coord):
    """
    This class implements a `Coord` for galactic longitude.
    """
    pass


class b(Coord):
    """
    This class implements a `Coord` for galactic latitude.
    """
    pass


# v = [l.from_sexagesimal("+156:18:00.421"), b.from_sexagesimal("+31:34:20.582")]
# print(*map(str, equatorial_to_galactic(*galactic_to_equatorial(*v))))
# print(*map(str, galactic_to_equatorial(*v)))
