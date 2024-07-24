import numpy as np

from src.statistics.stats_library.advanced_stats import acr_func_1d_cpp, acr_func_2d_cpp, str_func_cpp, increments_cpp

def autocorrelation_function(data: np.ndarray) -> np.ndarray:
    """
    Computes the one-dimensional autocorrelation function of a 2D array.

    Parameters
    ----------
    data : np.ndarray
        Data from which to compute the autocorrelation function.

    Returns
    -------
    autocorrelation_function : np.ndarray
        Two-dimensional array with every group of three elements representing the lag and its corresponding
        autocorrelation function and uncertainty.
    """
    return np.array(acr_func_1d_cpp(data))

def autocorrelation_function_2d(data: np.ndarray) -> np.ndarray:
    """
    Computes the two-dimensional autocorrelation function of a 2D array.

    Parameters
    ----------
    data : np.ndarray
        Data from which to compute the 2D autocorrelation function.

    Returns
    -------
    autocorrelation_function : np.ndarray
        Two-dimensional array with every group of three elements representing the x lag, the y lag and its corresponding
        autocorrelation function.
    """
    return np.array(acr_func_2d_cpp(data))

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
        Two-dimensional array with every group of three elements representing the lag and its corresponding structure
        function and uncertainty.
    """
    return np.array(str_func_cpp(data))

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
