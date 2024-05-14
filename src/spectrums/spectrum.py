from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.modeling import models, fitting
from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines


from src.headers.header import Header


class Spectrum():
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
        self.fitted_function = None

    def plot(self, title: str=None, fullscreen: bool=False, **kwargs):
        """
        Plots the spectrum along with the fit's residue, if a fit was made. If plotting the fit is desired, the 
        plot_fit() method should be used as it wraps this method in case of plotting fits.

        Parameters
        ----------
        title : str, default=None
            Text to be displayed as the title of the plot. This is used for debugging purposes.
        fullscreen : bool, default=False
            Specifies if the graph must be opened in fullscreen.
        kwargs : dict
            This argument may take any distribution to be plotted and is used to plot all the gaussian fits on the same
            plot. The name used for each keyword argument will be present in the plot's legend.
        """
        fig, axs = plt.subplots(2)
        axs[0].plot(self.arange(1, len(self.data) + 1), self.data, "g-", label="spectrum", linewidth=3, alpha=0.6)
        # for key, value in kwargs.items():
        #     if value is None:
        #         continue
        #     x_plot_gaussian = np.arange(1,self.x_values[-1]+0.05,0.05)
        #     if key == "fit":
        #         # Fitted entire function
        #         axs[0].plot(x_plot_gaussian*u.Jy, value(x_plot_gaussian*u.um), "r-", label=key)
        #     elif key == "subtracted_fit":
        #         # Residual distribution
        #         axs[1].plot(self.x_values, value, label=key)
        #     elif key == "NII":
        #         # NII gaussian
        #         axs[0].plot(x_plot_gaussian, value(x_plot_gaussian), "m-", label=key, linewidth="1")
        #     elif key == "NII_2":
        #         # Second NII gaussian
        #         axs[0].plot(x_plot_gaussian, value(x_plot_gaussian), "c-", label=key, linewidth="1")
        #     elif key == "SII1":
        #         # First SII gaussian
        #         axs[0].plot(x_plot_gaussian, value(x_plot_gaussian), "m-", label=key, linewidth="1")
        #     elif key == "SII2":
        #         # Second SII gaussian
        #         axs[0].plot(x_plot_gaussian, value(x_plot_gaussian), "c-", label=key, linewidth="1")
        #     elif key == "initial_guesses":
        #         # Simple points plotting
        #         axs[0].plot(value[0], value[1], "kx", label=key, markersize="10")
        #     else:
        #         # Fitted individual gaussians
        #         axs[0].plot(x_plot_gaussian, value(x_plot_gaussian), "y-", label=key, linewidth="1")
        
        axs[0].legend(loc="upper left", fontsize="7")
        axs[1].legend(loc="upper left", fontsize="7")
        plt.xlabel("channels")
        axs[0].set_ylabel("intensity")
        axs[1].set_ylabel("intensity")

        if title:
            fig.title(title)

        if fullscreen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        plt.show()

    def get_residue_stddev(self, bounds: tuple[int, int]=None) -> float:
        """
        Gets the standard deviation of the fit's residue.

        Parameters
        ----------
        bounds: tuple[int, int], default=None
            Bounds between which the residue's standard deviation should be calculated. If None is provided, the
            residue's stddev is calculated for all values. Bounds indexing is the same as lists, e.g. bounds=(0,2) gives
            x=1 and x=2.

        Returns
        -------
        residue's std : float
            Value of the residue's standard deviation.
        """
        if bounds is None:
            return np.std(self.get_subtracted_fit())#/u.Jy)
        else:
            return np.std(self.get_subtracted_fit()[slice(*bounds)])#/u.Jy)

    def get_subtracted_fit(self) -> np.ndarray:
        """
        Gets the subtracted fit's values.

        Returns
        -------
        subtracted fit : np.ndarray
            Result values of the gaussian fit subtracted to the y values.
        """
        subtracted_y = self.data - self.fitted_function(self.x_values*u.um)/u.Jy
        return subtracted_y

    def get_FWHM_channels(self, gaussian_function_index: int) -> np.ndarray:
        """
        Gets the full width at half maximum of a gaussian function along with its uncertainty in channels.

        Parameters
        ----------
        gaussian_function_index : str
            Index of the gaussian function whose FWHM in channels needs to be calculated.

        Returns
        -------
        FWHM : np.ndarray
            Array of the FWHM and its uncertainty measured in channels.
        """
        fwhm = 2*np.sqrt(2*np.log(2))*getattr(self.fitted_function, gaussian_function_index).stddev.value
        fwhm_uncertainty = 2*np.sqrt(2*np.log(2))*self.get_uncertainties()[gaussian_function_index]["stddev"]
        return np.array((fwhm, fwhm_uncertainty))

    def get_snr(self, gaussian_function_index: int) -> float:
        """
        Gets the signal to noise ratio of a peak. This is calculated as the amplitude of the peak divided by the
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
        return getattr(self.fitted_function, gaussian_function_index).amplitude / self.get_residue_stddev()#/u.Jy after amplitude

    def fit(self, parameter_bounds: dict) -> models:
        """
        Fits a Spectrum using the get_initial_guesses method and with parameter bounds. Also set the astropy model of
        the fitted gaussian to the variable self.fit.

        Parameters
        ----------
        parameter_bounds : dict
            Bounds of each gaussian (numbered keys) and corresponding dictionary of bounded parameters. For example,
            parameter_bounds = {0 : {"amplitude": (0, 8)*u.Jy, "stddev": (0, 1)*u.um, "mean": (20, 30)*u.um}}.
        
        Returns
        -------
        fit : models
            Model of the fitted Spectrum.
        """
        initial_guesses = self.get_initial_guesses()

        spectrum = Spectrum1D(flux=self.data*u.Jy, spectral_axis=np.arange(1, len(self.data) + 1)*u.um)
        gaussians = [
            models.Gaussian1D(
                amplitude=initial_guesses[i].get("amplitude"),
                mean=initial_guesses[i].get("mean"),
                stddev=initial_guesses[i].get("stddev"),
                bounds=parameter_bounds[i]
            ) for i in range(len(parameter_bounds))
        ]
        self.fitted_function = fit_lines(
            spectrum,
            sum(gaussians),
            fitter=fitting.LMLSQFitter(calc_uncertainties=True),
            get_fit_info=True,
            maxiter=1e4
        )

        return self.fitted_function
