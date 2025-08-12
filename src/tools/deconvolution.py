import numpy as np
import src.graphinglib as gl
from scipy.signal import fftconvolve
from typing import Literal
from tqdm import tqdm

from src.hdu.cubes.cube import Cube
from src.hdu.maps.map import Map


def normalize_signals(x: np.ndarray, method: Literal["integral", "max"] = "integral") -> np.ndarray:
    """
    Normalizes the given signal using the specified method.

    Parameters
    ----------
    x : np.ndarray
        The data cube containing the spectrums to be normalized. This should be a 3D array where the first dimension
        represents the spectral axis.
    method : Literal["integral", "max"], default="integral"
        The normalization method to use. If "integral", the signal is normalized by its integral. If "max", the signal
        is normalized by its maximum value.

    Returns
    -------
    np.ndarray
        The normalized signal.
    """
    if method == "integral":
        return x / (np.sum(x, axis=0, keepdims=True) + 1e-10)
    elif method == "max":
        return x / (np.max(x, axis=0, keepdims=True) + 1e-10)
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'integral' or 'max'.")

def estimate_centroids(data: np.ndarray) -> np.ndarray:
    """
    Estimates the centroids of the peaks in a given data cube using a quadratic function. This function is a more robust
    estimator than the `argmax` operator, but does not need the cost of fitting a model to the data.

    Parameters
    ----------
    data : np.ndarray
        The data cube containing the spectrums to be processed. This should be a 3D array where the first dimension
        represents the spectral axis.

    Returns
    -------
    np.ndarray
        An array of the estimated centroids for each spectrum in the data cube. The centroids are given following
        1-based indexing.
    """
    max_indices = np.argmax(data, axis=0)
    max_indices = np.clip(max_indices, 1, data.shape[0] - 2)  # clip to avoid index out of bounds
    y_1 = np.take_along_axis(data, max_indices[None, :, :] - 1, axis=0)[0]
    y_2 = np.take_along_axis(data, max_indices[None, :, :], axis=0)[0]
    y_3 = np.take_along_axis(data, max_indices[None, :, :] + 1, axis=0)[0]
    centroids = max_indices + 0.5 * (y_1 - y_3) / (y_1 - 2*y_2 + y_3 + 1e-10) + 1
    return centroids

def roll_spectrums(data: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    Roll each spectrum in a 3D data cube by its individual offset.

    Parameters
    ----------
    data : np.ndarray
        3D array with shape (spectral, y, x) containing the spectrums
    offsets : np.ndarray
        2D array with shape (y, x) containing the roll offset for each spectrum

    Returns
    -------
    np.ndarray
        3D array with each spectrum rolled by its corresponding offset
    """
    offsets = np.nan_to_num(offsets, nan=0, posinf=0, neginf=0).round().astype(int)

    spectral_indices = np.arange(data.shape[0])[:, None, None]
    y_indices = np.arange(data.shape[1])[None, :, None]
    x_indices = np.arange(data.shape[2])[None, None, :]

    rolled_spectral_indices = (spectral_indices - offsets[None, :, :]) % data.shape[0]
    rolled_data = data[rolled_spectral_indices, y_indices, x_indices]
    return rolled_data

def deconvolve_cube(
    data: np.ndarray | Cube,
    lsf: np.ndarray | Cube,
    lsf_centroids: np.ndarray | Map,
    n_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deconvolves the spectrums in the given data cube using the provided line spread function (LSF) via Richardson-Lucy
    deconvolution. This removes the instrumental response from the spectrums, allowing for a clearer view of the
    underlying signal. This function wraps all the necessary preprocessing steps, including centering and normalizing
    the spectrums before applying the deconvolution algorithm with the `richardson_lucy_deconvolution` function.

    Parameters
    ----------
    data : np.ndarray | Cube
        The data cube containing the spectrums to be deconvolved. This should be a 3D array where the first dimension
        represents the spectral axis.
    lsf : np.ndarray | Cube
        The line spread function (LSF) used for deconvolution given as a data cube. This corresponds to the
        instrumental response function of the Fabry-Pérot interferometer. The spectral axis should be the first
        dimension.
    lsf_centroids : np.ndarray | Map
        The centroids of the LSF spectrums. This is used to align the LSF spectrums before deconvolution. These
        should be obtained from fitting the LSF spectrums to a model for greater precision. The centroids should be
        given following 1-based indexing.
    n_iterations : int
        The number of iterations to perform in the Richardson-Lucy algorithm.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing three elements: (deconvolved_data, offsetted_data, offsetted_lsf). These are given in the
        same format as the input data, with shape (spectral, y, x).
    """
    data = data.data if isinstance(data, Cube) else data
    lsf = lsf.data if isinstance(lsf, Cube) else lsf
    lsf_centroids = lsf_centroids.data if isinstance(lsf_centroids, Map) else lsf_centroids

    data = normalize_signals(data)
    lsf = normalize_signals(lsf)

    lsf_center_offset = data.shape[0] // 2 - (lsf.argmax(axis=0) + 1)  # offset to center the LSF spectrums
    data_centroids = estimate_centroids(data)
    data_offset = lsf_centroids - data_centroids + lsf_center_offset

    # Roll LSF and data spectrums by their individual offsets
    offsetted_lsf = roll_spectrums(lsf, lsf_center_offset)
    offsetted_data = roll_spectrums(data, data_offset)

    deconvolved_data = richardson_lucy_deconvolution(offsetted_data, offsetted_lsf, n_iterations)
    deconvolved_data = normalize_signals(deconvolved_data)
    deconvolved_centroids = estimate_centroids(deconvolved_data)
    deconvolved_offset = deconvolved_centroids
    deconvolved_offset = data.shape[0] // 2 - deconvolved_centroids
    offsetted_deconvolved = roll_spectrums(deconvolved_data, deconvolved_offset)

    return offsetted_deconvolved, offsetted_data, offsetted_lsf

def richardson_lucy_deconvolution(data: np.ndarray, lsf: np.ndarray, n_iterations: int) -> np.ndarray:
    """
    Performs Richardson-Lucy deconvolution on the given data using the provided line spread function (LSF).

    Parameters
    ----------
    data : np.ndarray | Cube
        The data cube containing the spectrums to be deconvolved. This should be a 3D array where the first dimension
        represents the spectral axis.
    lsf : np.ndarray | Cube
        The line spread function (LSF) used for deconvolution given as a data cube. This corresponds to the
        instrumental response function of the Fabry-Pérot interferometer. The spectral axis should be the first
        dimension.
    n_iterations : int
        The number of iterations to perform in the Richardson-Lucy algorithm.

    Returns
    -------
    np.ndarray
        The deconvolved data.
    """
    limits = 1e-6, 1e3
    data = data.astype(np.float64)
    estimate = np.full_like(data, 0.5)
    reversed_lsf = lsf[::-1]

    for _ in tqdm(range(n_iterations)):
        convolution = fftconvolve(estimate, lsf, mode="same", axes=0)
        convolution = np.clip(convolution, limits[0], None)
        ratio = data / convolution
        correction = fftconvolve(ratio, reversed_lsf, mode="same", axes=0)
        estimate *= correction
        estimate = np.clip(estimate, 0, limits[1])
    return estimate

def get_deconvolution_error(
    data: np.ndarray,
    lsf: np.ndarray,
    deconvolved: np.ndarray,
    sampling_range: int = 5,
)-> tuple[np.ndarray, np.ndarray]:
    """
    Gives the deconvolution score for the given spectrum and deconvolved spectrum. This is calculated by reconvolving
    the deconvolved spectrum with the LSF and comparing it to the original spectrum. Only the data points
    ± sampling_range channels around the peak of the original spectrum are considered for the score.

    .. warning::
        The score must be calculated with aligned and normalized spectra. Use the output from `deconvolve_cube` to
        ensure proper alignment.

    Parameters
    ----------
    data : np.ndarray
        The original spectrum that was deconvolved.
    lsf : np.ndarray
        The line spread function (LSF) used for deconvolution. This corresponds to the instrumental response function
        of the Fabry-Pérot interferometer.
    deconvolved : np.ndarray
        The deconvolved spectrum. See the `deconvolve_spectrum` function.
    sampling_range : int, default=5
        The number of channels to consider around the peak of the original spectrum for the score calculation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two elements: (error, reconvolved). The error is the deconvolution score for each pixel,
        which is the mean squared error between the original spectrum and the reconvolved deconvolved spectrum. This
        helps identify spectrums with poor deconvolution quality. The reconvolved spectrum is the result of
        reconvolving the deconvolved spectrum with the LSF, which can be useful for visual confirmation.
    """
    peak_index = data.shape[0] // 2 - 1
    lower_limit, upper_limit = peak_index - sampling_range, peak_index + sampling_range

    reconvolved = fftconvolve(deconvolved, lsf, mode="same", axes=0)
    reconvolved = normalize_signals(reconvolved)

    data_centroids = estimate_centroids(data)
    reconvolved_centroids = estimate_centroids(reconvolved)
    reconvolved_offset = data_centroids - reconvolved_centroids
    offsetted_reconvolved = roll_spectrums(reconvolved, reconvolved_offset)

    error = np.mean((data[lower_limit:upper_limit, :, :] - offsetted_reconvolved[lower_limit:upper_limit, :, :]) ** 2,
                    axis=0)
    return error, offsetted_reconvolved
