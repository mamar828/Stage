import numpy as np

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
