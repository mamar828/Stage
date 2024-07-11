import numpy as np

from src.statistics.stats_library.advanced_stats import autocorrelation_function_cpp, structure_function_cpp, \
                                                        increments_cpp


def autocorrelation_function(data: np.ndarray) -> np.ndarray:
    """
    Computes the autocorrelation function of a 2D array.

    Parameters
    ----------
    data : np.ndarray
        Data from which to compute the autocorrelation function.

    Returns
    -------
    autocorrelation_function : np.ndarray
        Two-dimensional array with every pair of element representing the lag and its corresponding autocorrelation.
    """
    return np.array(autocorrelation_function_cpp(data))

def structure_function(data: np.ndarray) -> np.ndarray:
    """
    Computes the structure function of a 2D array.

    Parameters
    ----------
    data : np.ndarray
        Data from which to compute the structure function.

    Returns
    -------
    structure_function : np.ndarray
        Two-dimensional array with every pair of element representing the lag and its corresponding structure function.
    """
    return np.array(structure_function_cpp(data))

def increments(data: np.ndarray) -> dict:
    """
    Computes the increments of a 2D array.

    Parameters
    ----------
    data : np.ndarray
        Data from which to compute the increments.

    Returns
    -------
    increments : dict
        Every key is a lag and the corresponding value is the list of increments with this lag.
    """
    increments_dict = {}
    increments = increments_cpp(data)
    for increment in increments:
        increments_dict[increment[0]] = np.array(increment[1:])
    return increments_dict
