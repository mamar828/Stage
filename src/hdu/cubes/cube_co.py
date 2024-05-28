import numpy as np
from pathos.pools import ProcessPool
from tqdm import tqdm
import awkward as ak
from eztcolors import Colors as C

from src.hdu.cubes.cube import Cube
from src.hdu.cubes.worker import _worker_split
from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.maps.map import Map
from src.hdu.arrays.array_2d import Array2D
from src.hdu.tesseract import Tesseract


class CubeCO(Cube):
    """
    Encapsulates the methods specific to CO data cubes.
    """

    def fit(self, spectrum_parameters: dict=None) -> tuple[Map, Tesseract]:
        """
        Fits the CubeCO with a variable number of peaks.
        WARNING: Due to the use of the multiprocessing library, calls to this function NEED to be made inside a 
        condition state with the following phrasing:
        if __name__ == "__main__":
        This prevents the code to recursively create instances of itself that would eventually overload the CPUs.

        Parameters
        ----------
        spectrum_parameters: dict=None
            Parameters for initialization of the SpectrumCO objects. Supported parameters are "peak_prominence", 
            "peak_minimum_height_sigmas", "peak_minimum_distance", "peak_width", "noise_channels",
            "initial_guesses_binning" and "max_residue_sigmas" (See the SpectrumCO constructor for a definition of each
            parameter). If None, or if keys are not specified, the parameters will take the default values given in the
            SpectrumCO constructor. 

        Returns
        -------
        fit results : tuple[Map, Tesseract]
            Results of fitting the entire Cube. The first element is a Map representing the chi-square of the fit at
            each pixel. The second elemenent is a Tesseract containing the adjusted function's parameters at every
            pixel.
        """
        self.spectrum_parameters = spectrum_parameters
        with ProcessPool() as pool:
            print(f"{C.YELLOW}Number of processes used: {pool.nodes}{C.END}")
            progressbar = tqdm(
                desc="Fitting",
                total=self.data.shape[1],
                unit="fit",
                colour="BLUE"
            )
            imap_iterator = pool.imap(
                lambda i: _worker_split(self, i),
                range(self.data.shape[1])
            )

            results = []
            for result in imap_iterator:
                results.append(result)
                progressbar.update(1)
            # results is now a list of [y_shape, x_shape, (chi2, fit_results)]

        results_array = ak.Array(results)
        chi2_array = Map(
            data=Array2D(results_array[:,:,0,0,0]),
            header=self.header
        )
        fit_results = Tesseract(
            data=results_array[:,:,1],
            header=self.header
        )

        return chi2_array, fit_results

    def _worker(self, row: int) -> list[list[float, np.ndarray]]:
        """
        Fits a line of a Cube.

        Parameters
        ----------
        row : int
            Vertical coordinate of the row that needs to be fitted.

        Returns
        -------
        fit results : list[list[float, np.ndarray]]
            Results of fitting every pixel along the line. Each pixel presents another list : [chi2, fit results]. The
            fit results is the converted DataFrame of the spectrum, now in a numpy array of shape (n_components, 6)
            where n_components is the number of fitted gaussians. The numpy array's format is determined by the
            spectrum's fit_results DataFrame.
            Note : the float is actually nested inside two other lists to preserve the number of dimensions.
        """
        map_ = self[:,row,:]
        results = []
        # A list of invalid values is created for further use
        # Warning ! To ease slicing with ak.Arrays, the number of dimensions should be constant
        nans = [[[np.NAN]], [[np.NAN]]]
        for spectrum in map_:
            if np.all(np.isnan(spectrum.data)):
                # Empty spectrum
                results.append(nans)

            else:
                spectrum = spectrum.upgrade(SpectrumCO)
                if self.spectrum_parameters is not None:
                    for param, value in self.spectrum_parameters.items():
                        setattr(spectrum, param.upper(), value)

                spectrum.fit()

                if spectrum.is_successfully_fitted:
                    results.append([
                        [[spectrum.get_fit_chi2()]],
                        spectrum.fit_results.to_numpy()
                    ])
                else:
                    results.append(nans)

        return results
