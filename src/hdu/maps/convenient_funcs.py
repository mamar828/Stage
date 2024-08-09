import numpy as np
import scipy
import scipy.constants

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
    speed_map = ((channel_map - src_cube.header["CRPIX3"])*src_cube.header["CDELT3"] + src_cube.header["CRVAL3"]) / 1000
    return speed_map

def get_kinetic_temperature(
        amplitude_map: Map
) -> Map:
    """
    Computes the kinetic temperature of a given amplitude map.
    """
    T_kin = 5.53 / np.log(1 + 5.53/(amplitude_map + 0.148))
    return T_kin

def get_13co_column_density(
        fwhm_13co: Map,
        kinetic_temperature_13co: Map,
        kinetic_temperature_12co: Map,
) -> Map:
    """
    Computes the 13CO column density from the given FWHM and kinetic temperature maps.
    """
    µ = 0.112           # taken from https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds10.pdf
    nu = 110.20e9       # taken from https://tinyurl.com/23e45pj3
    print(fwhm_13co[19,27])
    print(kinetic_temperature_13co[19,27])
    print(kinetic_temperature_12co[19,27])
    column_density = (
        3 * scipy.constants.h * fwhm_13co * kinetic_temperature_13co 
        / (4 * np.pi**3 * μ**2 * (1 - np.exp(- scipy.constants.h*nu / (scipy.constants.k*kinetic_temperature_12co))))
    ) * 1e48
    return column_density
