import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.tools.zurflueh_filter.cpp_lib.zfilter import zfilter_cpp


def create_zfilter(pixel_width: int=13) -> np.ndarray:
    """
    Creates a Zurflueh filter of a given width by interpolating a known 33x33 kernel.

    Parameters
    ----------
    pixel_width : int, default=13
        Width of the filter, in pixels, that is used for convolution. This number should be odd and not greater than 33
        as the interpolation of a 33x33 filter is used.

    Returns
    -------
    filter : np.ndarray
        Two-dimensional array of a normalized filter of the required width.
    """
    if pixel_width % 2 == 0:
        raise ValueError("Pixel width must be odd.")

    coefficients_33x33 = np.loadtxt("src/tools/zurflueh_filter/cpp_lib/coefficients_33x33.csv", delimiter=",")
    linspace = np.linspace(-16,16,33)
    interp = RegularGridInterpolator((linspace, linspace), coefficients_33x33)

    half_width = pixel_width // 2
    new_linspace = np.arange(pixel_width) - half_width
    xx, yy = np.meshgrid(new_linspace, new_linspace)
    new_filter = interp(np.array(np.stack((xx, yy), axis=-1)) * (16 / half_width))   # Interpolate on a spreaded grid
    new_filter /= new_filter.sum()
    return new_filter

def zfilter(data: np.ndarray=None, pixel_width: int=13) -> np.ndarray:
    """
    Computes the Zurflueh filter of a 2D array.

    Parameters
    ----------
    data : np.ndarray
        Data from which to compute the Zurflueh filter.
    pixel_width : int, default=13
        Width of the filter, in pixels, that is used for convolution. This number should be odd and not greater than 33
        as the interpolation of a 33x33 filter is used.

    Returns
    -------
    filtered data : np.ndarray
        Two-dimensional filtered array with the same shape as the input array.
    """
    return np.array(zfilter_cpp(data, create_zfilter(pixel_width)))
