from __future__ import annotations
import numpy as np
from graphinglib import Curve, Scatter, Figure, MultiFigure, FitFromPolynomial
import pandas as pd
from copy import deepcopy
from astropy import units as u
from astropy.modeling import models, fitting, CompoundModel
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines

from src.headers.header import Header


class Spectrum:
    """
    Encapsulate all the methods of any data cube's spectrum.
    """

    def __init__(self, data: np.ndarray, header: Header):
        """
        Initializes a Spectrum object with a certain header, whose spectral information will be taken.

        Parameters
        ----------
        data : np.ndarray
            Detected intensity at each channel.
        header : Header
            Allows for the calculation of the FWHM using the header's informations.
        """
        self.data = data
        self.header = header
        self.initial_guesses = {}
        self.fitted_function: models.Gaussian1D | CompoundModel = None
        self.fit_results: pd.DataFrame = None

    def __len__(self) -> int:
        return len(self.data)
    
    def copy(self) -> Spectrum:
        return deepcopy(self)
    
    @property
    def isnan(self) -> bool:
        return np.all(np.isnan(self.data))

    def setattrs(self, attributes: dict):
        for key, value in attributes.items():
            setattr(self, key.upper(), value)

    @staticmethod
    def fit_needed(func):
        # Decorator to handle exceptions when a fit has not been made 
        def inner_func(self, *args, **kwargs):
            if self.fitted_function:
                return func(self, *args, **kwargs)
            else:
                return None
        return inner_func

    @property
    def x_values(self) -> np.ndarray:
        """
        Gives the x values associated with the Spectrum's data.
        
        Returns
        -------
        x_values : np.ndarray
            Range from 1 and has the same length than the data array. The start value is chosen to match with SAOImage
            ds9 and with the headers, whose axes start at 1.
        """
        return np.arange(1, len(self) + 1)

    @property
    @fit_needed
    def predicted_data(self) -> np.ndarray:
        """
        Gives the y values predicted by the fit in the form of data points.
        
        Returns
        -------
        predicted_data : np.ndarray
            Array representing the predicted intensity at every channel. The first element corresponds to channel 1.
        """
        return self.fitted_function(self.x_values * u.um) / u.Jy
    
    @property
    def is_successfully_fitted(self) -> bool:
        """
        Outputs whether the fit succeeded.
        
        Returns
        -------
        success : bool
            True if the fit succeeded, False otherwise.
        """
        fit_state = True if self.fitted_function is not None else False
        return fit_state and not self.fit_results.empty

    @property
    def plot(self) -> Curve:
        """
        Gives the plot of the spectrum with a Curve.

        Returns
        -------
        spectrum : Curve
            Curve representing the spectrum's values at every channel.
        """
        curve = Curve(
            x_data=self.x_values,
            y_data=self.data,
            label="Spectrum"
        )
        return curve

    @property
    @fit_needed
    def individual_functions_plot(self) -> tuple[Curve]:
        """
        Gives the plot of every fitted functions with a single or many Curves. These Curves may be unpacked and given to
        the Figure.add_elements method.

        Returns
        -------
        fitted function : tuple[Curve]
            Curves representing every function fitted to the spectrum's values at every channel.
        """
        curves = [
            Curve(
                x_data=self.x_values,
                y_data=models.Gaussian1D(
                    amplitude=self.fit_results.amplitude.value[i],
                    mean=self.fit_results["mean"].value[i],
                    stddev=self.fit_results.stddev.value[i]
                )(self.x_values),
                label=f"Gaussian {i}"
            ) for i in self.fit_results.index
        ]
        return tuple(curves)

    @property
    @fit_needed
    def total_functions_plot(self) -> Curve:
        """
        Gives the total plot the global fitted function with a single Curve

        Returns
        -------
        global function : Curve
            Curve representing the sum of every gaussian fitted.
        """
        total = sum(self.individual_functions_plot)
        total.label = "Sum"
        return total

    @property
    @fit_needed
    def initial_guesses_plot(self) -> Scatter:
        """
        Gives the spectrum's initial guesses with a Scatter.

        Returns
        -------
        initial guesses : Scatter
            Scatter giving the spectrum's initial guesses for every detected peak.
        """
        initial_guesses_array = np.array([
            [peak["mean"], peak["amplitude"]] for peak in self.initial_guesses.values()
        ])
        scatter = Scatter(
            x_data=initial_guesses_array[:,0],
            y_data=initial_guesses_array[:,1],
            label="Initial guesses",
            marker_size=50,
            marker_style="v",
            face_color="black"
        )
        return scatter

    @property
    @fit_needed
    def residue_plot(self) -> Curve:
        """
        Gives the plot of the fit's residue with a Curve.

        Returns
        -------
        residue : Curve
            Curve representing the fit's residue at every channel.
        """
        curve = Curve(
            x_data=self.x_values,
            y_data=self.get_subtracted_fit(),
            label="Residue"
        )
        return curve

    def auto_plot(self):
        """
        Plots automatically a Spectrum in a preprogrammed way. The shown Figure will present on one axis the spectrum
        and on the other the residue if a fit was made.
        """
        plot_elements = [self.plot]
        if self.fitted_function is None:
            multi_figure = MultiFigure(1, 1, size=(10, 7))
        else:
            multi_figure = MultiFigure(2, 1, size=(10, 7))
            [plot_elements.append(element) for element in self.individual_functions_plot]
            plot_elements.append(self.initial_guesses_plot)
            figure_2 = Figure(
                x_label="Channels",
                y_label="Residue"
            )
            figure_2.add_elements(self.residue_plot)
            multi_figure.add_figure(figure_2, 1, 0, 1, 1)
        
        figure_1 = Figure(
            x_label="Channels",
            y_label="Intensity"
        )
        figure_1.add_elements(*plot_elements)
        multi_figure.add_figure(figure_1, 0, 0, 1, 1)
        multi_figure.show()

    def bin(self, bin: int) -> Spectrum:
        """
        Bins a Spectrum.

        Parameters
        ----------
        bin : int
            Number of channels to be binned together. A value of 2 would mean that the number of channels will be
            divided by two and each new channel will represent the mean of two previous channels.

        Returns
        -------
        spectrum : Spectrum
            Binned Spectrum.
        """
        cropped_pixels = np.array(self.data.shape) % np.array(bin)
        data_copy = self.data[:self.data.shape[0] - cropped_pixels[0]]

        reshaped_data = data_copy.reshape((data_copy.shape[0] // bin, bin))
        data_copy = np.mean(reshaped_data, axis=1)
        return self.__class__(data_copy, self.header.bin([bin]))

    def fit(self, parameter_bounds: dict):
        """
        Fits a Spectrum using the get_initial_guesses method and with parameter bounds. Also sets the astropy model of
        the fitted gaussians to the variable self.fitted_function.

        Parameters
        ----------
        parameter_bounds : dict
            Bounds of every parameter for every gaussian. 
            Example : {"amplitude": (0, 8)*u.Jy, "stddev": (0, 1)*u.um, "mean": (20, 30)*u.um}.
        """
        initial_guesses = self.get_initial_guesses()
        if initial_guesses:
            spectrum = Spectrum1D(flux=self.data*u.Jy, spectral_axis=self.x_values*u.um)
            gaussians = [
                models.Gaussian1D(
                    amplitude=initial_guesses[i]["amplitude"]*u.Jy,
                    mean=initial_guesses[i]["mean"]*u.um,
                    stddev=initial_guesses[i]["stddev"]*u.um,
                    bounds=parameter_bounds
                ) for i in range(len(initial_guesses))
            ]
            self.fitted_function = fit_lines(
                spectrum,
                sum(gaussians, models.Gaussian1D(amplitude=0, mean=0, stddev=0)), # Null element is needed to init sum
                fitter=fitting.LMLSQFitter(calc_uncertainties=True),
                get_fit_info=True,
                maxiter=int(1e4)
            )
            self._store_fit_results()
    
    def polyfit(self, degree: int=3) -> FitFromPolynomial:
        """
        Fits the spectrum using a polynomial of a given degree. This method is used for correcting continuum shifts.

        Parameters
        ----------
        degree : int, optional
            Degree of the polynomial to fit. Defaults to 3.

        Returns
        -------
        polyfit_function : FitFromPolynomial
            Fitted polynomial function in the form of a FitFromPolynomial.
        """
        return FitFromPolynomial(self.plot, degree)

    def _store_fit_results(self):
        """
        Stores the results of the fit in the fit_results variable in the forme of a DataFrame.
        """
        values, uncertainties = self._get_cleaned_fitted_function_data()

        title = np.repeat(["amplitude", "mean", "stddev"], 2)
        subtitle = np.array(["value", "uncertainty"]*3)
        data = np.vstack((values, uncertainties)).T.reshape(len(values) // 3, 6)
        df = pd.DataFrame(zip(title, subtitle, data), columns=["title", "subtitle", "data"])
        df.set_index(["title", "subtitle"], inplace=True)

        self.fit_results = pd.DataFrame(data=data, columns=pd.MultiIndex.from_tuples(zip(title, subtitle)))
    
    def _get_cleaned_fitted_function_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Gives the values and uncertainties of each parameter of self.fitted_function for every non zero gaussian. This
        is used to clean the gaussians outputted by the fit which may sometimes be null.

        Returns
        -------
        data : tuple[np.ndarray, np.ndarray]
            Value and uncertainty of every relevant parameter of self.fitted_function.
        """
        n_submodels = self.fitted_function.n_submodels
        values = self.fitted_function.parameters.reshape((n_submodels, 3))
        uncertainties = np.sqrt(np.diag(self.fitted_function.meta["fit_info"]["param_cov"])).reshape((n_submodels, 3))
        mask = np.ones_like(values, dtype=bool)
        
        for i, model in enumerate(values):
            amplitude, mean, stddev = model
            if amplitude < 1e-4 or stddev < 1e-4:
                mask[i] = False

        filtered_flat_values = values[mask].flatten()
        filtered_flat_uncertainties = uncertainties[mask].flatten()
        return filtered_flat_values, filtered_flat_uncertainties

    @fit_needed
    def get_residue_stddev(self, bounds: slice=None) -> float:
        """
        Gives the standard deviation of the fit's residue.

        Parameters
        ----------
        bounds: slice, default=None
            Bounds between which the residue's standard deviation should be calculated. If None is provided, the
            residue's stddev is calculated for all values. Bounds indexing is the same as lists, e.g. bounds=slice(0,2)
            gives x=1 and x=2.

        Returns
        -------
        residue's stddev : float
            Value of the residue's standard deviation.
        """
        if bounds is None:
            stddev = np.std(self.get_subtracted_fit())
        else:
            stddev = np.std(self.get_subtracted_fit()[bounds])
        return stddev

    @fit_needed
    def get_subtracted_fit(self) -> np.ndarray:
        """
        Gives the subtracted fit's values.

        Returns
        -------
        subtracted fit : np.ndarray
            Result values of the gaussian fit subtracted to the y values.
        """
        subtracted_y = self.data - self.predicted_data
        return subtracted_y

    @fit_needed
    def get_FWHM_channels(self, gaussian_function_index: int) -> np.ndarray:
        """
        Gives the full width at half maximum of a gaussian function along with its uncertainty in channels.

        Parameters
        ----------
        gaussian_function_index : str
            Index of the gaussian function whose FWHM in channels needs to be calculated.

        Returns
        -------
        FWHM : np.ndarray
            Array of the FWHM and its uncertainty measured in channels.
        """
        stddev = np.array([self.fit_results.stddev.value[gaussian_function_index],
                           self.fit_results.stddev.uncertainty[gaussian_function_index]])
        fwhm = 2 * np.sqrt(2*np.log(2)) * stddev
        return fwhm

    @fit_needed
    def get_snr(self, gaussian_function_index: int) -> float:
        """
        Gives the signal to noise ratio of a peak. This is calculated as the amplitude of the peak divided by the
        residue's standard deviation.
    
        Parameters
        ----------
        gaussian_function_index : str
            Index of the gaussian function whose amplitude will be used to calculate the snr.

        Returns
        -------
        snr : float
            Value of the signal to noise ratio.
        """
        return self.fit_results.amplitude.value[gaussian_function_index] / self.get_residue_stddev()

    @fit_needed    
    def get_fit_chi2(self) -> float:
        """
        Gives the chi-square of the fit.

        Returns
        -------
        chi2 : float
            Chi-square of the fit.
        """
        chi2 = np.sum(self.get_subtracted_fit()**2 / np.var(self.data[self.NOISE_CHANNELS]))
        normalized_chi2 = chi2 / len(self)
        return float(normalized_chi2)
