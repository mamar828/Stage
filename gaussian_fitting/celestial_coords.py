class Celestial_coords():
    # This class is inherited by the following two classes and allows using of isinstance() in 
    # fits_analyzer.Fits_file.set_wcs() method
    pass



class RA(Celestial_coords):
    """
    Encapsulate the methods specific to right ascension coordinates.
    """

    def __init__(self, time: str):
        """
        Initialize a RA object.

        Arguments
        ---------
        time: str. Specifies the right ascension in clock format (HH:MM:SS.SSS -> Hours, Minutes, Seconds)
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



class DEC(Celestial_coords):
    """
    Encapsulate the methods specific to declination coordinates.
    """

    def __init__(self, time: str):
        """
        Initialize a DEC object.

        Arguments
        ---------
        time: str. Specifies the declination in clock format (DD:MM:SS.SSS -> Degrees, Minutes, Seconds)
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
