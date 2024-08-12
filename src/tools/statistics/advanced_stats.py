import numpy as np
from scipy.ndimage import gaussian_filter
from graphinglib import Contour

from src.tools.statistics.stats_library.advanced_stats import (
    acr_func_1d_cpp, acr_func_2d_cpp, str_func_cpp, increments_cpp
)

def autocorrelation_function(data: np.ndarray) -> np.ndarray:
    """
    Computes the one-dimensional autocorrelation function of a 2D array. The intermediate estimator is used and the
    values are normalized with the value at zero lag.

    Parameters
    ----------
    data : np.ndarray
        Data from which to compute the autocorrelation function.

    Returns
    -------
    autocorrelation_function : np.ndarray
        Two-dimensional array with every group of two elements representing the lag and its corresponding
        autocorrelation function and uncertainty.
    """
    return np.array(acr_func_1d_cpp(data))

def autocorrelation_function_2d(data: np.ndarray) -> np.ndarray:
    """
    Computes the two-dimensional autocorrelation function of a 2D array. The intermediate estimator is used and the
    values are normalized with the value at zero lag.

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

def get_autocorrelation_function_2d_contour(autocorrelation_function_2d: np.ndarray) -> Contour:
    """
    Reads the output given by the autocorrelation_function_2d function and translates to a Contour object. A 3x3
    gaussian filter is used for smoothing the data.

    Parameters
    ----------
    autocorrelation_function_2d : np.ndarray
        Two-dimensional array with every group of three elements representing the x lag, the y lag and its corresponding
        autocorrelation function. The output of the autocorrelation_function_2d function may be given.

    Returns
    -------
    contour plot : Contour
        A Contour object which correctly represents the x and y grid as well as the z data, which has been smoothed with
        a 3x3 gaussian filter.
    """
    # Copy paste the data with a diagonal reflection
    data = np.append(
        autocorrelation_function_2d,
        autocorrelation_function_2d * np.tile((-1, -1, 1), (autocorrelation_function_2d.shape[0], 1)),
        axis=0
    )

    x_lim = np.min(data[:,0]), np.max(data[:,0])
    y_lim = np.min(data[:,1]), np.max(data[:,1])

    x_grid, y_grid = np.meshgrid(np.arange(x_lim[0], x_lim[1] + 1), 
                                 np.arange(y_lim[0], y_lim[1] + 1))

    z_data = np.zeros_like(x_grid)
    for x, y, z in data:
        z_data[int(y-np.min(data[:,1])), int(x-np.min(data[:,0]))] = z
    z_data = gaussian_filter(z_data, 3)

    contour = Contour(
        x_mesh=x_grid,
        y_mesh=y_grid,
        z_data=z_data,
        show_color_bar=True,
        number_of_levels=list(np.arange(-1, 1 + 0.1, 0.1)),
        filled=False,
        color_map="viridis",
    )
    return contour
