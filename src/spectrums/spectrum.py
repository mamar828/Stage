from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy
from astropy import units as u
from astropy.modeling import models, fitting, CompoundModel
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines
from matplotlib.axes import Axes

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
    
    @classmethod
    def from_map(cls, map) -> Spectrum:
        """
        Creates a Spectrum from a Map.
        
        Parameters
        ----------
        map : Map
            Map to load the Spectrum from. This must be a previously sliced Map with now one dimension, but with the
            header of a Map (2 dimensions).
        
        Returns
        -------
        spectrum : Spectrum
            One-dimensional Spectrum of the 1D Map.
        """
        header_naxes = [map.header[f"NAXIS{i}"] for i in range(map.header["NAXIS"], 0, -1)]
        missing_axis = header_naxes.index((set(header_naxes) - set(map.data.shape)).pop())
        spectrum = cls(
            data=map.data,
            header=map.header.flatten(axis=missing_axis)
        )
        return spectrum
    
    @classmethod
    def from_cube(cls, cube) -> Spectrum:
        """
        Creates a Spectrum from a Cube.
        
        Parameters
        ----------
        cube : Cube
            Cube to load the Spectrum from. This must be a previously sliced Cube with now one dimension, but with the
            header of a Cube (3 dimensions).
        
        Returns
        -------
        spectrum : Spectrum
            One-dimensional Spectrum of the 1D Cube.
        """
        new_header = cube.header
        for i in range(2):
            header_naxes = [new_header[f"NAXIS{i}"] for i in range(new_header["NAXIS"], 0, -1)]
            missing_axis = header_naxes.index((set(header_naxes) - set(cube.data.shape)).pop())
            new_header = new_header.flatten(axis=missing_axis)

        spectrum = cls(
            data=cube.data,
            header=new_header
        )
        return spectrum

    def upgrade(self, cls: Spectrum) -> Spectrum:
        """
        Constructs an upper-level Spectrum object from a given spectrum.
        
        Parameters
        ----------
        cls : Spectrum-inherited class
            Class of a Spectrum that inherits from this base Spectrum class.

        Returns
        -------
        spectrum : cls
            Converted spectrum
        """
        spectrum = cls(
            data=self.data.copy(),
            header=self.header.copy()
        )
        return spectrum

    def copy(self) -> Spectrum:
        return deepcopy(self)
    
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
        state = True if self.fitted_function is not None else False
        return state

    def plot(self, ax: Axes, **kwargs):
        """
        Plots the spectrum on an axis.

        Parameters
        ----------
        ax : Axes
            Axis on which to plot the Spectrum.
        kwargs : dict
            This argument may take any distribution to be plotted and is used to plot all the gaussian fits on the same
            plot. The name used for each keyword argument will be present in the plot's legend. The keyword "fit" plots
            the values in red and the keyword "initial_guesses" plots the values as a point scatter.
        """
        ax.plot(self.x_values, self.data, "k-", label="spectrum", linewidth=1, alpha=1)
        for key, value in kwargs.items():
            if value is not None:
                if key == "fit":
                    # Fitted entire function
                    ax.plot(self.x_values*u.Jy, value(self.x_values*u.um), "r-", label=key)
                elif key == "initial_guesses":
                    # Simple points plotting
                    ax.plot(value[:,0], value[:,1], "bv", label=key, markersize="4", alpha=0.5)
                else:
                    # Fitted individual gaussians
                    ax.plot(self.x_values, value(self.x_values), "y-", label=key, linewidth="1")
        
        ax.legend(loc="upper left", fontsize="7")

    def plot_residue(self, ax: Axes):
        """
        Plots the fit's residue.

        Parameters
        ----------
        ax : Axes
            Axis on which to plot the fit's residue
        """
        if self.fitted_function:
            ax.plot(self.get_subtracted_fit())

    def auto_plot(self, text: str=None, fullscreen: bool=False):
        """
        Plots automatically a Spectrum in a preprogrammed way. The shown Figure will present on one axis the spectrum
        and on the other the residue if a fit was made.

        Parameters
        ----------
        text : str, default=None
            Text to be displayed as the title of the plot. This is used for debugging purposes.
        fullscreen : bool, default=False
            Specifies if the graph must be opened in fullscreen.  
        """
        if self.fitted_function is None:
            fig, ax = plt.subplots(1)
            self.plot(ax)
        else:
            fig, axs = plt.subplots(2)
            self.plot(axs[0], fit=self.fitted_function)
            self.plot_residue(axs[1])
        
        fig.supxlabel("Channels")
        fig.supylabel("Intensity")

        if text:
            fig.suptitle(text)

        if fullscreen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        plt.show()

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
            Bounds of each gaussian (numbered keys) and corresponding dictionary of bounded parameters. For example,
            parameter_bounds = {0 : {"amplitude": (0, 8)*u.Jy, "stddev": (0, 1)*u.um, "mean": (20, 30)*u.um}}.
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
                self.sum_gaussians(gaussians),
                fitter=fitting.LMLSQFitter(calc_uncertainties=True),
                get_fit_info=True,
                maxiter=int(1e4)
            )
            self._store_fit_results()
    
    @staticmethod
    def sum_gaussians(gaussians: list) -> CompoundModel:
        """
        Sums a list of models.Gaussian1D objects.

        Parameters
        ----------
        gaussians : list
            List of Gaussian1D objects to sum together.

        Returns
        -------
        function : CompoundModel
            Model representing the sum of all gaussians.
        """
        total = gaussians[0]
        if len(gaussians) >= 1:
            for gaussian in gaussians[1:]:
                total += gaussian
        return total
    
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
        Gives the values and uncertainties of each parameter of self.fitted_function for every non zero gaussian.

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
