import numpy as np
import src.graphinglib as gl
from scipy.signal import convolve
from typing import Literal

from src.hdu.cubes.cube import Cube


def preprocess_signal(x: np.ndarray, normalize_method: Literal["integral", "max"] = "integral") -> np.ndarray:
    """
    Recenters signals in a data cube and normalizes them.

    Parameters
    ----------
    x : np.ndarray
        The data cube containing the spectrums to be preprocessed. This should be a 3D array where the first dimension
        represents the spectral axis.
    normalize_method : Literal["integral", "max"], default="integral"
        The method used for normalization. If "integral", the signal is normalized by its integral. If "max", the signal
        is normalized by its maximum value.
    """
    centroids = 1 + np.argmax(x, axis=0)
    shifts = int(np.round(x.shape[0] // 2 - centroids + 1))
    recentered = np.roll(x, shifts, axis=0)
    if normalize_method == "integral":
        normalized = recentered / np.sum(recentered, axis=0, keepdims=True)
    elif normalize_method == "max":
        normalized = recentered / np.max(recentered, axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown normalization method: {normalize_method}. Use 'integral' or 'max'.")
    return normalized

def deconvolve_cube(data: np.ndarray | Cube, lsf: np.ndarray | Cube, n_iterations: int) -> np.ndarray | Cube:
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
    n_iterations : int
        The number of iterations to perform in the Richardson-Lucy algorithm.

    Returns
    -------
    np.ndarray | Cube
        The data cube containing the deconvolved spectrums, which is the result of applying Richardson-Lucy
        deconvolution to each spectrum using their associated LSF.
    """
    input_spectrums = data.data if isinstance(data, Cube) else data
    input_spectrums = preprocess_signal(input_spectrums)
    lsf_spectrums = lsf.data if isinstance(lsf, Cube) else lsf
    lsf_spectrums = preprocess_signal(lsf_spectrums)

    output_signal = richardson_lucy_deconvolution(input_spectrums, lsf_spectrums, n_iterations)
    output_signal = preprocess_signal(output_signal)
    if isinstance(data, Cube) and isinstance(lsf, Cube):
        output_signal = Cube(output_signal, header=data.header)

    return output_signal

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

    def convolve_3d(signal, response):
        result = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            for j in range(signal.shape[2]):
                result[:, i, j] = convolve(signal[:, i, j], response[:, i, j], mode="same")
        return result

    for _ in range(n_iterations):
        convolution = convolve_3d(estimate, lsf)
        convolution = np.clip(convolution, limits[0], None)
        ratio = data / convolution
        correction = convolve_3d(ratio, reversed_lsf)
        estimate *= correction
        estimate = np.clip(estimate, 0, limits[1])
    return estimate

def get_deconvolution_error(
    spectrum: np.ndarray,
    deconvolved: np.ndarray,
    lsf: np.ndarray,
    sampling_range: int = 7
)-> float:
    """
    Gives the deconvolution score for the given spectrum and deconvolved spectrum. This is calculated by reconvolving
    the deconvolved spectrum with the LSF and comparing it to the original spectrum. Only the data points
    ± sampling_range channels around the peak of the original spectrum are considered for the score.

    Parameters
    ----------
    spectrum : np.ndarray
        The original spectrum that was deconvolved.
    deconvolved : np.ndarray
        The deconvolved spectrum. See the `deconvolve_spectrum` function.
    lsf : np.ndarray
        The line spread function (LSF) used for deconvolution. This corresponds to the instrumental response function
        of the Fabry-Pérot interferometer.
    sampling_range : int, optional
        The number of channels to consider around the peak of the original spectrum for the score calculation.

    Returns
    -------
    float
        The deconvolution score, which is the mean squared error between the original spectrum and the reconvolved
        deconvolved spectrum.
    """
    centered_spectrum = preprocess_signal(spectrum, normalize_method="max")
    centered_deconvolved = preprocess_signal(deconvolved)
    centered_lsf = preprocess_signal(lsf)

    reconvolved = convolve(centered_deconvolved, centered_lsf, mode="same")
    reconvolved = preprocess_signal(reconvolved, normalize_method="max")

    peak_index = np.argmax(centered_spectrum)
    start_index = max(0, peak_index - sampling_range)
    end_index = min(len(centered_spectrum), peak_index + sampling_range + 1)

    score = np.mean((centered_spectrum[start_index:end_index] - reconvolved[start_index:end_index]) ** 2)
    # gl.SmartFigure(elements=[
    #     gl.Curve(np.arange(len(centered_spectrum[:,0,0])), centered_spectrum[:,0,0], label="Original Spectrum"),
    #     gl.Curve(np.arange(len(reconvolved[:,0,0])), reconvolved[:,0,0], label="Reconvolved Deconvolved Spectrum")
    # ]).show()
    return score
