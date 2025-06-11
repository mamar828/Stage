from __future__ import annotations
import numpy as np
import scipy as sp
from typing import Self, Literal
from pathos.pools import ProcessPool
from pathos.helpers import cpu_count
from tqdm import tqdm
from astropy.modeling import models

from src.hdu.fits_file import FitsFile
from src.hdu.arrays.array_3d import Array3D
from src.hdu.cubes.cube import Cube
from src.hdu.tesseract import Tesseract
from src.tools.array_functions import list_to_array
from src.tools.messaging import notify_function_end


class FittableCube(Cube):
    """
    This class implements a fittable data cube that contains generic methods for finding initial guesses and fitting
    data.
    """

    def find_peaks_gaussian_estimates(self, voigt: bool = False, **kwargs) -> Self:
        """
        Finds gaussian initial guesses using scipy.signal's find_peaks algorithm as well as the peak_widths algorithm.
        These initial guesses can then be used to fit gaussian or voigt functions to the data.

        Parameters
        ----------
        voigt : bool, default=False
            If True, the initial guesses will be made for a Voigt profile instead of a Gaussian profile. This results in
            four parameters per model: amplitude_L, x_0, fwhm_L, fwhm_G.

        kwargs : Any
            Arguments to pass to the scipy.signal.find_peaks function. Useful parameters include:
            - `height`: Required height of the peaks.
            - `threshold`: Required threshold of peaks, the vertical distance to its neighboring samples.
            - `distance`: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
            - `prominence`: Required prominence of peaks.
            - `width`: Required width of peaks in samples.
            See the documentation of `scipy.signal.find_peaks` for more details.

        Returns
        -------
        Self
            Cube with the initial guesses for the specified model. The guesses are stored along the first axis, ordered
            as (in the case of `voigt=False`): amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, ..., where the
            first three values are the parameters of the first Gaussian model, the next three are the parameters of the
            second Gaussian model, and so on. For `voigt=True`, the parameters are given in groups of four:
            lorenzian amplitude, mean, fwhm and gaussian fwhm.
        """
        transposed_data = self.flatten_3d_array(self.data)
        peak_means = [sp.signal.find_peaks(spectrum, **kwargs)[0] for spectrum in transposed_data]
        peak_amplitudes = [spectrum[peaks] for spectrum, peaks in zip(transposed_data, peak_means)]
        peak_widths = [sp.signal.peak_widths(spectrum, peaks)[0]
                       for spectrum, peaks in zip(transposed_data, peak_means)]
        peak_means = list_to_array(peak_means)
        peak_amplitudes = list_to_array(peak_amplitudes)
        peak_widths = list_to_array(peak_widths)
        assert peak_means.size > 0, \
            "No peaks were detected in the data. Please check the parameters passed to find_peaks."

        peak_means += 1     # correct for the 0-based indexing in numpy but 1-based indexing in the data

        if voigt:
            fwhms_L = 0.7 * peak_widths
            fwhms_G = 0.3 * peak_widths

            # Find the amplitude correction factor to take into account the Voigt profile shape
            non_nan_mask = ~np.isnan(peak_means)
            amplitude_correction = peak_amplitudes[~np.isnan(peak_means)][0] / models.Voigt1D().evaluate(
                peak_means[non_nan_mask][0],
                peak_means[non_nan_mask][0],
                peak_amplitudes[non_nan_mask][0],
                fwhms_L[non_nan_mask][0],
                fwhms_G[non_nan_mask][0],
            )
            # reshape to (n_data, n_models, 4)
            guesses = np.dstack((peak_amplitudes * amplitude_correction, peak_means, fwhms_L, fwhms_G))
        else:
            # reshape to (n_data, n_models, 3)
            guesses = np.dstack((peak_amplitudes, peak_means, peak_widths / (2*np.sqrt(2*np.log(2)))))

        guesses = guesses.reshape(self.data.shape[2], self.data.shape[1], -1)
        guesses = guesses.T

        return self.__class__(guesses, self.header)

    @notify_function_end
    def fit(
        self,
        model,
        guesses: Cube | Array3D,
        number_of_parameters: int = 3,
        number_of_tasks: int | Literal["auto"] = "auto",
        **kwargs,
    ) -> Tesseract:
        """
        Fits a model to the Cube data. This function wraps the `scipy.optimize.curve_fit` function and for an entire
        Cube, and uses multiprocessing to speed up the fitting process.


        Parameters
        ----------
        model : callable
            The model to fit to the data. This must be a callable function with the signature:
            `model(x, *params)`, where `x` is the independent variable and `params` are the parameters to fit. The
            number of parameters must match number of parameters given in `guesses`.

            .. warning::
                For now, this function only supports fitting a single model to the entire Cube. The resulting Tesseract
                will therefore always have a single model index.

        guesses : Cube | Array3D
            Initial guesses for the parameters of the model. If None, the function will try to find initial guesses. The
            guesses must be given along the first axis, ordered as:
            amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, ..., where the first three values are the
            parameters of the first Gaussian model, the next three are the parameters of the second Gaussian model, and
            so on. The output of the `find_peaks_gaussian_estimates` method can be used as is.
        number_of_parameters : int, default=3
            Number of parameters in the model. This is used to reshape the output of the fitting process. The default
            value is 3, which corresponds to a Gaussian model with amplitude, mean, and standard deviation parameters.
        number_of_tasks : int | Literal["auto"], default="auto"
            Number of tasks to split the fitting process into. If "auto", it will be set to ten times the number of CPU
            cores available on the system.
        kwargs : Any
            Additional arguments to pass to the fitting function.

        Returns
        -------
        Tesseract
            Results of the fitting process.
        """
        guesses_array = guesses.data if isinstance(guesses, Cube) else guesses
        if number_of_tasks == "auto":
            number_of_tasks = cpu_count() * 10
        x_values = np.arange(self.shape[0]) + 1
        nan_array = np.full(guesses_array.shape[0] * 2, np.nan)

        # @FitsFile.silence_function
        def worker_fit_spectrums(spectrums, guesses):
            results = []
            for spectrum_i, guesses_i in zip(spectrums, guesses):
                valid_guesses = guesses_i[~np.isnan(guesses_i)] # remove NaN values
                if valid_guesses.size == 0:
                    results.append(nan_array)
                else:
                    try:
                        params, pcov = sp.optimize.curve_fit(
                            f=model,
                            xdata=x_values,
                            ydata=spectrum_i,
                            p0=valid_guesses.flatten(),
                            maxfev=kwargs.get("maxfev", 10000),
                        )
                    except RuntimeError:
                        results.append(nan_array)
                        continue

                    perr = np.sqrt(np.diag(pcov))
                    results.append(np.pad(
                        np.column_stack((params, perr)).flatten(),
                        (0, nan_array.size - 2*params.size),
                        mode='constant',
                        constant_values=np.nan,
                    ))
            return results

        data_2d, guesses_2d = self.flatten_3d_array(self.data), self.flatten_3d_array(guesses_array)
        splitted_data = np.array_split(data_2d, number_of_tasks)
        splitted_guesses = np.array_split(guesses_2d, number_of_tasks)

        packed_arguments = [
            (chunk_data, chunk_guesses)
            for chunk_data, chunk_guesses in zip(splitted_data, splitted_guesses)
            if chunk_data.size > 0 and chunk_guesses.size > 0
        ]

        fit_params_chunks = []
        pbar = tqdm(total=len(packed_arguments), desc="Fitting", unit="chunk", colour="blue", miniters=1)
        with ProcessPool() as pool:
            for result in pool.imap(lambda args: worker_fit_spectrums(*args), packed_arguments):
                fit_params_chunks.extend(result)
                pbar.update(1)
                pbar.refresh()

        fit_params = np.array(fit_params_chunks)
        fit_params = fit_params.reshape(self.data.shape[2], self.data.shape[1], -1).T
        fit_params = fit_params.reshape(-1, 2 * number_of_parameters, self.data.shape[1], self.data.shape[2])
        fit_params = fit_params.swapaxes(0, 1)

        tesseract_header = self.header.flatten(0)
        tesseract_header["CTYPE3"] = "model index"
        tesseract_header["CTYPE4"] = "param1 + unc., param2 + unc., param3 + unc., ..."

        fit_results = Tesseract(
            data=fit_params,
            header=tesseract_header,
        )

        return fit_results
