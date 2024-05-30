import numpy as np

from src.hdu.cubes.cube import Cube
from src.hdu.maps.map import Map


def get_FWHM(stddev_map: Map, source_cube: Cube) -> Map:
    fwhm = stddev_map * (2 * np.sqrt(2*np.log(2)) * np.abs(source_cube.header["CDELT3"]) / 1000)
    return fwhm
