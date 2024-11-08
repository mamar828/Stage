import numpy as np
import graphinglib as gl
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from functools import partial
from copy import deepcopy
from uncertainties import ufloat

from src.tools.statistics.stats_library.stats_library import (
    acr_func_1d_kleiner_dickman_cpp,
    acr_func_1d_boily_cpp,
    acr_func_2d_kleiner_dickman_cpp,
    acr_func_2d_boily_cpp,
    str_func_cpp,
    increments_cpp
)
from src.tools.statistics.split_normal import SplitNormal


np_sort = lambda arr: arr[np.argsort(arr[:,0])]

def autocorrelation_function(data: np.ndarray, method: str="Boily") -> np.ndarray:
    """
    Computes the one-dimensional autocorrelation function of a 2D array. The intermediate estimator is used and the
    values are normalized with the value at zero lag.

    Parameters
    ----------
    data : np.ndarray
        Data from which to compute the autocorrelation function.
    method : str, default="Boily"
        Method to use for autocorrelation function calculation. The two available methods are "Boily" and
        "Kleiner Dickman". The Boily method simply averages without any normalizing factor whilst the Kleiner Dickman
        method uses a normalization factor dependent on the number of points.

    Returns
    -------
    autocorrelation_function : np.ndarray
        Two-dimensional array. If method="Boily" every group of three elements represents the lag and its corresponding
        autocorrelation function and uncertainty. If method="Kleiner Dickman" every group of two elements represents
        the lag and its corresponding autocorrelation function, without uncertainty. The returned array is sorted
        according to the lag value.
    """
    if method == "Boily":
        return np_sort(np.array(acr_func_1d_boily_cpp(deepcopy(data))))
    elif method == "Kleiner Dickman":
        return np_sort(np.array(acr_func_1d_kleiner_dickman_cpp(deepcopy(data))))
    else:
        raise ValueError(f"Unsupported autocorrelation function method: {method}")

def autocorrelation_function_2d(data: np.ndarray, method: str="Boily") -> np.ndarray:
    """
    Computes the two-dimensional autocorrelation function of a 2D array. The intermediate estimator is used and the
    values are normalized with the value at zero lag.

    Parameters
    ----------
    data : np.ndarray
        Data from which to compute the 2D autocorrelation function.
    method : str, default="Boily"
        Method to use for autocorrelation function calculation. The two available methods are "Boily" and
        "Kleiner Dickman". The Boily method simply averages without any normalizing factor whilst the Kleiner Dickman
        method uses a normalization factor dependent on the number of points.

    Returns
    -------
    autocorrelation_function : np.ndarray
        Two-dimensional array with every group of three elements representing the x lag, the y lag and its corresponding
        autocorrelation function.
    """
    if method == "Boily":
        return np.array(acr_func_2d_boily_cpp(deepcopy(data)))
    elif method == "Kleiner Dickman":
        return np.array(acr_func_2d_kleiner_dickman_cpp(deepcopy(data)))
    else:
        raise ValueError(f"Unsupported autocorrelation function method: {method}")

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
        function and uncertainty. The returned array is sorted according to the lag value.
    """
    return np_sort(np.array(str_func_cpp(deepcopy(data))))

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
    increments = increments_cpp(deepcopy(data))
    for increment in increments:
        increments_dict[increment[0]] = np.array(increment[1:])
    return increments_dict

def evaluate_delta_f2(data: np.ndarray) -> float:
    """
    Evaluates the ∆F_2(tau_0) parameter which quantifies the Zurflueh filter's efficiency that implies quasi-homogeneous
    motions.

    Parameters
    ----------
    data : np.ndarray
        Data from which to compute the ∆F_2(tau_0) parameter.

    Returns
    -------
    delta_f2_tau_0 : float
        ∆F_2(tau_0) parameter.
    """
    # Find tau_0
    autocorrelation_1d = autocorrelation_function(data, "Boily")
    autocorrelation_1d_curve = gl.Curve(autocorrelation_1d[:,0], autocorrelation_1d[:,1])
    zero_intersects = autocorrelation_1d_curve.create_intersection_points(
        gl.Curve(
            [autocorrelation_1d_curve.x_data],
            [0]*len(autocorrelation_1d_curve.x_data)
        )
    )
    if len(zero_intersects) > 0:
        tau_0 = zero_intersects[0].x

        # Compute ∆F_2(tau_0)
        structure_func = structure_function(data)
        mask = (structure_func[:,0] <= tau_0)
        if structure_func[mask,0].shape[0] > 1:
            linear = lambda x, a, b: a*x + b
            a, b = curve_fit(
                linear,
                np.log10(structure_func[mask,0]),
                np.log10(structure_func[mask,1]),
                [0.3, 0.5],
                maxfev=100000
            )[0]
            F_2_fit_tau_0 = linear(tau_0, a, b)

            # Compute F_1(0) (≈ 1)
            F_1_0 = autocorrelation_1d[autocorrelation_1d[:,0] == 0, 1]

            delta_f2_tau_0 = np.abs(F_2_fit_tau_0 - 2*F_1_0)
            return float(delta_f2_tau_0)

def get_autocorrelation_function_scatter(autocorrelation_function_data: np.ndarray) -> gl.Scatter:
    """
    Reads the output given by the autocorrelation_function function and translates it to a gl.Scatter object.

    Parameters
    ----------
    autocorrelation_function_data : np.ndarray
        Two-dimensional array with every group of three elements representing the lag and its corresponding value and
        uncertainty. The output of the autocorrelation_function function may be given.

    Returns
    -------
    scatter plot : gl.Scatter
        A Scatter object which correctly represents the lag as well as the value and uncertainty of the autocorrelation
        function.
    """
    scat = gl.Scatter(
        x_data=autocorrelation_function_data[:,0],
        y_data=autocorrelation_function_data[:,1],
        face_color="black",
        marker_size=3
    )
    scat.add_errorbars(
        y_error=autocorrelation_function_data[:,2],
        cap_width=None,
        errorbars_line_width=0.25
    )
    return scat

def get_autocorrelation_function_2d_contour(autocorrelation_function_2d_data: np.ndarray) -> gl.Contour:
    """
    Reads the output given by the autocorrelation_function_2d function and translates it to a gl.Contour object. A 3x3
    gaussian filter is used for smoothing the data.

    Parameters
    ----------
    autocorrelation_function_2d_data : np.ndarray
        Two-dimensional array with every group of three elements representing the x lag, the y lag and its corresponding
        autocorrelation function. The output of the autocorrelation_function_2d function may be given.

    Returns
    -------
    contour plot : gl.Contour
        A Contour object which correctly represents the x and y grid as well as the z data, which has been smoothed with
        a 3x3 gaussian filter.
    """
    # Copy paste the data with a diagonal reflection
    data = np.append(
        autocorrelation_function_2d_data,
        autocorrelation_function_2d_data * np.tile((-1, -1, 1), (autocorrelation_function_2d_data.shape[0], 1)),
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

    contour = gl.Contour(
        x_mesh=x_grid,
        y_mesh=y_grid,
        z_data=z_data,
        show_color_bar=True,
        number_of_levels=list(np.arange(-1, 1 + 0.1, 0.1)),
        filled=False,
        color_map="viridis",
    )
    return contour

def get_fitted_structure_function_figure(
        data: np.ndarray, 
        fit_bounds: tuple[float, float],
        number_of_iterations: int=10000
) -> gl.Figure:
    """
    Gives the figure of a fitted structure function in the given interval, computing the fit using Monte-Carlo
    uncertainties. The log10 of the data is taken and a linear fit is computed.

    Parameters
    ----------
    data : np.ndarray
        Data from which to compute the structure function. This should be the data outputted by the function "structure
        function".
    fit_bounds : tuple[float, float]
        x interval in which to execute the linear fit. This should exclude the first few points and the points until
        decorrelation, i.e. where the curve is not linear anymore.
    number_of_iterations : int
        Number of Monte-Carlo iterations to compute the fit uncertainty.

    Returns
    -------
    figure : gl.Figure
        A log-log Figure containing the data points and their uncertainty as well as a linear fit in the given bounds
        with its corresponding equation.
    """
    logged_data = np.log10(data)
    scatter = gl.Scatter(
        logged_data[:,0],
        logged_data[:,1],
        marker_size=3,
        face_color="black",
    )
    # Uncertainties are given in the order left, right or bottom, top
    uncertainties = np.array([
        np.abs(logged_data[:,1] - np.log10(data[:,1] - data[:,2])),
        np.abs(np.log10(data[:,1] + data[:,2]) - logged_data[:,1]),
    ])
    scatter.add_errorbars(
        y_error=uncertainties,
        cap_width=0,
        errorbars_line_width=0.25,
    )

    # Fit and its uncertainty
    m = (fit_bounds[0] < logged_data[:,0]) & (logged_data[:,0] < fit_bounds[1])     # generate the fit mask
    data_distributions = [SplitNormal(loc, *u) for loc, u in zip(logged_data[m,1], uncertainties.T[m])]
    values = np.array([sn.random(number_of_iterations) for sn in data_distributions]).T
    parameters = []
    for val in values:
        parameters.append(curve_fit(
            f=lambda x, m, b: m*x + b,
            xdata=logged_data[m,0],
            ydata=val,
            p0=[0.1,0.1],
            maxfev=100000
        )[0])

    parameters = np.array(parameters)
    m, b = parameters.mean(axis=0)
    dm, db = parameters.std(axis=0)     # uncertainties on the m and b parameters
    slope = ufloat(m, dm)
    fit = gl.Curve.from_function(
        lambda x: m*x + b,
        *fit_bounds,
        color="red",
        label=f"Slope : {slope:.1u}".replace("+/-", " ± "),
        line_width=1,
        number_of_points=2
    )
    y_fit_errors = db + np.array(fit_bounds)*dm
    fit.add_error_curves(
        y_error=y_fit_errors,
        error_curves_line_style=""
    )

    fig = gl.Figure(x_lim=(0,1.35), y_lim=(-0.2,0.5))
    fig.add_elements(scatter, fit)
    return fig
