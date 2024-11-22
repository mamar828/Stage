import numpy as np
import scipy
import scipy.constants as c
import warnings

from src.hdu.cubes.cube import Cube
from src.hdu.maps.map import Map


def get_FWHM(
        stddev_map: Map,
        src_cube: Cube
) -> Map:
    """ 
    Converts a stddev map into a FWHM map.
    """
    fwhm = stddev_map * (2 * np.sqrt(2*np.log(2)) * np.abs(src_cube.header["CDELT3"]) / 1000)
    return fwhm

def get_speed(
        channel_map: Map,
        src_cube: Cube
) -> Map:
    """
    Converts a channel map into a speed map.
    """
    warnings.warn("The output of this function has been modified by a 1000 factor.")
    speed_map = (channel_map - src_cube.header["CRPIX3"])*src_cube.header["CDELT3"] + src_cube.header["CRVAL3"]
    return speed_map

def get_kinetic_temperature(
        amplitude_map: Map
) -> Map:
    """
    Computes the kinetic temperature of a given amplitude map. Note that the kinetic temperature is assumed to be equal
    to the excitation temperature.
    """
    T_kin = 5.532 / np.log(1 + 1 / (amplitude_map/5.532 + 0.151))
    return T_kin

def integrate_gaussian(
        amplitude_map: Map,
        stddev_map: Map,
) -> Map:
    """
    Calculates the gaussian's area under the curve from -∞ to ∞.
    """
    # As the error function is odd, the integral is calculated as twice the function evaluated at high x
    # The error function converges to 1 for x -> ∞
    area = 2 * amplitude_map * stddev_map * np.sqrt(np.pi / 2)
    return area

def get_13co_column_density(
        stddev_13co: Map,
        antenna_temperature_13co: Map,
        antenna_temperature_12co: Map,
) -> Map:
    """
    Computes the 13CO column density from the given FWHM and temperature maps.

    Parameters
    ----------
    stddev_13co : Map
        Map of the 13CO emission's standard deviation, in km/s.
    antenna_temperature_13co : Map
        Map of the 13CO emission's amplitude, which corresponds to the antenna temperature, in K. Note that that the
        amplitude must first be divided by 0.43 for correction and that this method assumes this correction has already
        been applied.
    antenna_temperature_12co : Map
        Map of the 12CO emission's amplitude, which corresponds to the antenna temperature, in K., which is assumed to
        be equal to the excitation temperature, in km/s.

    Returns
    -------
    column_density : Map
        Map of the calculated 13CO column density, in cm^{-2}.
    """
    nu = 110.20e9       # taken from https://tinyurl.com/23e45pj3
    A_10 = 6.294e-8     # taken from https://home.strw.leidenuniv.nl/~moldata/datafiles/13co.dat
    g_0 = 2*0+1
    g_1 = 2*1+1
    T_rad = 2.725
    T_x = get_kinetic_temperature(antenna_temperature_12co)
    column_density = integrate_gaussian(
        amplitude_map=antenna_temperature_13co,
        stddev_map=stddev_13co
    ) * (0.8 * (g_0/(g_1*A_10)) * ((c.pi*c.k*nu**2)/(c.h*c.c**3)) * 
        1 / (
            (1 / (np.exp((c.h*nu)/(c.k*T_x))-1)
           - 1 / (np.exp((c.h*nu)/(c.k*T_rad))-1))
          * (1 - np.exp(-(c.h*nu)/(c.k*T_x)))
        )
    )
    return column_density
