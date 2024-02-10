from astropy.io import fits


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
        self.hours, self.minutes, self.seconds = [float(element) for element in time.split(":")]
    
    def to_deg(self) -> float:
        """
        Convert the time to degrees.

        Returns
        -------
        float: converted time in degrees.
        """
        return (self.hours*3600 + self.minutes*60 + self.seconds)/(24*3600) * 360

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
        self.angle, self.minutes, self.seconds = [float(element) for element in time.split(":")]
    
    def to_deg(self) -> float:
        """
        Convert the time to degrees.

        Returns
        -------
        float: converted time in degrees.
        """
        return self.angle + (self.minutes*60 + self.seconds)/(3600)



class Galactic_coords(Celestial_coords):
    # This class is inherited by the following two classes and allows using of isinstance()
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
        self.angle, self.minutes, self.seconds = [float(element) for element in coord.split(":")]
    
    def to_deg(self) -> float:
        """
        Convert the coord to degrees.

        Returns
        -------
        float: converted coord in degrees.
        """
        return self.angle + (self.minutes*60 + self.seconds)/(3600)
        return (self.hours*3600 + self.minutes*60 + self.seconds)/(24*3600) * 360

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
            return round((self.to_deg() - header["CRVAL1"]) / header["CDELT1"] + header["CRPIX1"])
        elif "GLON" in header["CTYPE2"]:
            print(self.to_deg())
            return round((self.to_deg() - header["CRVAL2"]) / header["CDELT2"] + header["CRPIX2"])
        elif "GLON" in header["CTYPE3"]:
            return round((self.to_deg() - header["CRVAL3"]) / header["CDELT3"] + header["CRPIX3"])
        else:
            raise ValueError("Header does not have GLON type")


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
        self.angle, self.minutes, self.seconds = [float(element) for element in coord.split(":")]
    
    def __str__(self):
        return f"{self.angle:.0f}:{self.minutes:.0f}:{self.seconds:06.3f}"
    
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
            t = str((pixel - header["CRPIX1"]) * header["CDELT1"] + header["CRVAL1"]).split(".")
        elif "GLAT" in header["CTYPE2"]:
            t = str((pixel - header["CRPIX2"]) * header["CDELT2"] + header["CRVAL2"]).split(".")
        elif "GLAT" in header["CTYPE3"]:
            t = str((pixel - header["CRPIX3"]) * header["CDELT3"] + header["CRVAL3"]).split(".")
        else:
            raise ValueError("Header does not have GLAT type")
        
        dec = float("0." + t[1])
        return cls(f"{t[0]}:{dec*3600//60}:{dec*3600%60}")

    def to_deg(self) -> float:
        """
        Convert the time to degrees.

        Returns
        -------
        float: converted time in degrees.
        """
        return self.angle + (self.minutes*60 + self.seconds)/(3600)

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
            return round((self.to_deg() - header["CRVAL1"]) / header["CDELT1"] + header["CRPIX1"])
        elif "GLAT" in header["CTYPE2"]:
            return round((self.to_deg() - header["CRVAL2"]) / header["CDELT2"] + header["CRPIX2"])
        elif "GLAT" in header["CTYPE3"]:
            return round((self.to_deg() - header["CRVAL3"]) / header["CDELT3"] + header["CRPIX3"])
        else:
            raise ValueError("Header does not have GLAT type")
