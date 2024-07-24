import numpy as np

from src.tools.zurflueh_filter.cpp_lib.zfilter import zfilter_cpp


def zfilter(data: np.ndarray) -> np.ndarray:
    """
    Computes the Zurflueh filter of a 2D array.

    Parameters
    ----------
    data : np.ndarray
        Data from which to compute the Zurflueh filter.

    Returns
    -------
    filtered data : np.ndarray
        Two-dimensional filtered array with the same shape as the input array.
    """
    return np.array(zfilter_cpp(data))
