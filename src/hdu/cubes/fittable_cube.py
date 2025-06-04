from __future__ import annotations
import numpy as np
import scipy as sp
from typing import Self, Literal
from pathos.pools import ProcessPool
from pathos.helpers import cpu_count
from tqdm import tqdm

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
        Finds gaussian initial guesses using a find_peaks algorithm. These initial guesses can then be used to fit
        gaussian functions to the data.

        Parameters
        ----------
        voigt : bool, default=False
            If True, the initial guesses will be made for a Voigt profile instead of a Gaussian profile. This very
            simple option simply duplicates the stddev parameter to give four parameters for the Voigt profile.

            .. note::
                Voigt profiles are typically defined by the FWHM of the Lorentzian and Gaussian components, but this
                method uses twice the standard deviation of the Gaussian component as a proxy for the lorentzian FWHM.
                This is a very rough approximation and may not always yield accurate results.

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
            Cube with the initial guesses for the Gaussian model. The guesses are stored along the first axis, ordered
            as: amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, ..., where the first three values are the
            parameters of the first Gaussian model, the next three are the parameters of the second Gaussian model, and
            so on.
        """
        transposed_data = self.flatten_3d_array(self.data)
        peak_means = [sp.signal.find_peaks(spectrum, **kwargs)[0] for spectrum in transposed_data]
        peak_amplitudes = [spectrum[peaks] for spectrum, peaks in zip(transposed_data, peak_means)]
        peak_means = list_to_array(peak_means)
        peak_amplitudes = list_to_array(peak_amplitudes)
        assert peak_means.size > 0, \
            "No peaks were detected in the data. Please check the parameters passed to find_peaks."

        # Estimate stddevs
        peak_stddevs = []
        for means, amplitude in zip(peak_means.T, peak_amplitudes.T):    # iterate over each detected peak
            half_max_difference = transposed_data - amplitude[:,None] / 2
            half_max_intersect_mask = np.abs(np.diff(np.sign(half_max_difference))).astype(bool)
            intersects_x = [np.where(mask)[0] + 1 for mask in half_max_intersect_mask]

            current_stddevs = []
            for intersect, mean in zip(intersects_x, means):
                if np.isnan(mean):
                    current_stddevs.append(np.nan)
                else:
                    lower_bound_candidates = intersect[intersect < mean]
                    lower_bound = lower_bound_candidates.max() if len(lower_bound_candidates) > 0 else 0
                    upper_bound_candidates = intersect[intersect > mean]
                    upper_bound = upper_bound_candidates.min() if len(upper_bound_candidates) > 0 else 0
                    current_stddevs.append((upper_bound - lower_bound) / (2*np.sqrt(2*np.log(2))))

            peak_stddevs.append(current_stddevs)

        peak_stddevs = np.array(peak_stddevs).T
        peak_means += 1     # correct for the 0-based indexing in numpy but 1-based indexing in the data

        # Combine the results into a single array and reshape it
        if voigt:
            guesses = np.dstack((peak_amplitudes, peak_means, peak_stddevs * 2, peak_stddevs))
        else:
            guesses = np.dstack((peak_amplitudes, peak_means, peak_stddevs))        # shape is (n_data, n_models, 3)

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
