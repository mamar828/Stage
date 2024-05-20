from __future__ import annotations
from astropy.io import fits
import numpy as np


class Celestial_coords():
    pass


class Equatorial_coords(Celestial_coords):
    pass

class RA(Equatorial_coords):
    """
    Encapsulate the methods specific to right ascension coordinates.
    """

    def __init__(self, time: str):
        """
        Initialize a RA object.

        Arguments
        ---------
        time: str. Specifies the right ascension in clock format (HH:MM:SS.SSS -> Hours, Minutes, Seconds).
        """
        # Split and convert each element in the time str to floats
        hours, minutes, seconds = [float(element) for element in time.split(":")]
        self.sexagesimal = (hours*3600 + minutes*60 + seconds)/(24*3600) * 360
    
    @classmethod
    def from_sexagesimal(cls, sexagesimal: float) -> RA:
        """
        Create an object from the sexagesimal value directly.
        
        Arguments
        ---------
        sexagesimal: float. Value in sexagesimal form.

        Returns
        -------
        RA object: object with the specified sexagesimal value.
        """
        new_object = cls("0:0:0")
        new_object.sexagesimal = sexagesimal
        return new_object

    def __str__(self) -> float:
        return str(self.sexagesimal)


class DEC(Equatorial_coords):
    """
    Encapsulate the methods specific to declination coordinates.
    """

    def __init__(self, time: str):
        """
        Initialize a DEC object.

        Arguments
        ---------
        time: str. Specifies the declination in clock format (DD:MM:SS.SSS -> Degrees, Minutes, Seconds).
        """
        # Split and convert each element in the time str to floats
        angle, minutes, seconds = [float(element) for element in time.split(":")]
        self.sexagesimal = angle + (minutes*60 + seconds)/(3600)
    
    @classmethod
    def from_sexagesimal(cls, sexagesimal: float) -> RA:
        """
        Create an object from the sexagesimal value directly.
        
        Arguments
        ---------
        sexagesimal: float. Value in sexagesimal form.

        Returns
        -------
        DEC object: object with the specified sexagesimal value.
        """
        new_object = cls("0:0:0")
        new_object.sexagesimal = sexagesimal
        return new_object

    def __str__(self) -> float:
        return str(self.sexagesimal)



class Galactic_coords(Celestial_coords):
    pass


class l(Galactic_coords):
    """
    Encapsulate the methods specific to galactic longitude coordinates.
    """

    def __init__(self, coord: str):
        """
        Initialize a RA object.

        Arguments
        ---------
        coord: str. Specifies the galactic longitude in clock format (DDD:MM:SS.SSS -> Degrees, Minutes, Seconds).
        """
        # Split and convert each element in the coord str to floats
        angle, minutes, seconds = [float(element) for element in coord.split(":")]
        self.sexagesimal = angle + (minutes*60 + seconds)/(3600)
    
    @classmethod
    def from_sexagesimal(cls, sexagesimal: float) -> RA:
        """
        Create an object from the sexagesimal value directly.
        
        Arguments
        ---------
        sexagesimal: float. Value in sexagesimal form.

        Returns
        -------
        l object: object with the specified sexagesimal value.
        """
        new_object = cls("0:0:0")
        new_object.sexagesimal = sexagesimal
        return new_object
    
    @classmethod
    def from_equatorial(cls, RA_object: RA, DEC_object: DEC):
        """ 
        Equation from https://www.atnf.csiro.au/people/Tobias.Westmeier/tools_coords.php.
        """
        alpha = RA_object.sexagesimal
        alpha_0 = 192.8595
        delta = DEC_object.sexagesimal
        delta_0 = 27.1284
        l_0 = 122.9320
        return cls.from_sexagesimal(l_0-np.arctan(
            (np.cos(delta)*np.sin(alpha-alpha_0)) / (
                np.sin(delta)*np.cos(delta_0)-np.cos(delta)*np.sin(delta_0)*np.cos(alpha-alpha_0)))
        )

    def __str__(self) -> float:
        return str(self.sexagesimal)

    def to_pixel(self, header: fits.Header) -> int:
        """
        Convert the object to a pixel number using the header

        Arguments
        ---------
        header: astropy.io.fits.Header

        Returns
        -------
        int: rounded pixel.
        """
        if "GLON" in header["CTYPE1"]:
            return round((self.sexagesimal - header["CRVAL1"]) / header["CDELT1"] + header["CRPIX1"])
        elif "GLON" in header["CTYPE2"]:
            return round((self.sexagesimal - header["CRVAL2"]) / header["CDELT2"] + header["CRPIX2"])
        elif "GLON" in header["CTYPE3"]:
            return round((self.sexagesimal - header["CRVAL3"]) / header["CDELT3"] + header["CRPIX3"])
        else:
            raise ValueError("Header does not have GLON type")
    
    def to_clock(self) -> str:
        """
        Display the object in the form of a string (DDD:MM:SS.SSS -> Degrees, Minutes, Seconds).

        Returns
        -------
        str: clock-formatted object
        """
        sep = str(self.sexagesimal).split(".")
        return f"{sep[0]}:{int(60*float(f'0.{sep[1]}')//1):00}:{60*float(f'0.{sep[1]}')%1*60:06.3f}"


class b(Galactic_coords):
    """
    Encapsulate the methods specific to galactic latitude coordinates.
    """

    def __init__(self, coord: str):
        """
        Initialize a DEC object.

        Arguments
        ---------
        time: str. Specifies the galactic latitude in clock format (DD:MM:SS.SSS -> Degrees, Minutes, Seconds).
        """
        # Split and convert each element in the coord str to floats
        angle, minutes, seconds = [float(element) for element in coord.split(":")]
        self.sexagesimal = angle + (minutes*60 + seconds)/(3600)
    
    @classmethod
    def from_sexagesimal(cls, sexagesimal: float) -> RA:
        """
        Create an object from the sexagesimal value directly.
        
        Arguments
        ---------
        sexagesimal: float. Value in sexagesimal form.

        Returns
        -------
        b object: object with the specified sexagesimal value.
        """
        new_object = cls("0:0:0")
        new_object.sexagesimal = sexagesimal
        return new_object
    
    @classmethod
    def from_pixel(cls, pixel: int, header: fits.Header):
        """
        Create a galactic latitude object from a pixel number.
        
        Arguments
        ---------
        pixel: int. Number of the pixel.
        header: astropy.io.fits.Header. Header to use for conversion.
        
        Returns
        -------
        b object: galactic latitude object with the corresponding data.
        """
        if "GLAT" in header["CTYPE1"]:
            sexagesimal = (pixel - header["CRPIX1"]) * header["CDELT1"] + header["CRVAL1"]
        elif "GLAT" in header["CTYPE2"]:
            sexagesimal = (pixel - header["CRPIX2"]) * header["CDELT2"] + header["CRVAL2"]
        elif "GLAT" in header["CTYPE3"]:
            sexagesimal = (pixel - header["CRPIX3"]) * header["CDELT3"] + header["CRVAL3"]
        else:
            raise ValueError("Header does not have GLAT type")
        
        return cls.from_sexagesimal(sexagesimal)
    
    @classmethod
    def from_equatorial(cls, RA_object: RA, DEC_object: DEC):
        """ 
        Equation from https://www.atnf.csiro.au/people/Tobias.Westmeier/tools_coords.php.
        """
        alpha = RA_object.sexagesimal
        alpha_0 = 192.8595
        delta = DEC_object.sexagesimal
        delta_0 = 27.1284
        return cls.from_sexagesimal(
            np.arcsin(np.sin(delta)*np.sin(delta_0)+np.cos(delta)*np.cos(delta_0)*np.cos(alpha-alpha_0))
        )

    def __str__(self) -> float:
        return str(self.sexagesimal)

    def to_pixel(self, header: fits.Header) -> int:
        """
        Convert the object to a pixel number using the header

        Arguments
        ---------
        header: astropy.io.fits.Header

        Returns
        -------
        int: rounded pixel.
        """
        if "GLAT" in header["CTYPE1"]:
            return round((self.sexagesimal - header["CRVAL1"]) / header["CDELT1"] + header["CRPIX1"])
        elif "GLAT" in header["CTYPE2"]:
            return round((self.sexagesimal - header["CRVAL2"]) / header["CDELT2"] + header["CRPIX2"])
        elif "GLAT" in header["CTYPE3"]:
            return round((self.sexagesimal - header["CRVAL3"]) / header["CDELT3"] + header["CRPIX3"])
        else:
            raise ValueError("Header does not have GLAT type")
    
    def to_clock(self) -> str:
        """
        Display the object in the form of a string (DD:MM:SS.SSS -> Degrees, Minutes, Seconds).

        Returns
        -------
        str: clock-formatted object
        """
        sep = str(self.sexagesimal).split(".")
        return f"{sep[0]}:{int(60*float(f'0.{sep[1]}')//1):02d}:{60*float(f'0.{sep[1]}')%1*60:06.3f}"
