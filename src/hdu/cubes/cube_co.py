import numpy as np
from pathos.pools import ProcessPool
from tqdm import tqdm
import awkward as ak
from eztcolors import Colors as C

from src.hdu.cubes.cube import Cube
from src.hdu.cubes.worker import _worker_split
from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.arrays.array_2d import Array2D


class CubeCO(Cube):
    """
    Encapsulates the methods specific to CO data cubes.
    """

    def fit(self, spectrum_parameters: dict=None) -> None:
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
            "peak_minimum_height_sigmas", "peak_minimum_distance" and "noise_channels". If None, or if keys are not
            specified, the parameters will take the default values given in the SpectrumCO constructor.

        Returns
        -------

        """
        import dill
        # with open("fit4.pkl", "rb") as f:
        #     dilly = dill.load(f)
        # """"""""" 
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
            # """""""""

        def extract_awkward(array: ak.Array, slices: tuple):
            # Function to ease the use of awkward arrays
            new_array = []
            for row in array:
                for sub_array in row:
                    # try:
                    #     np.all(np.isnan(sub_array))
                    # except:
                    #     print(sub_array)
                    if np.NAN in sub_array:
                        new_array.append([np.nan for _ in slices])
                    else:
                        try:
                            new_array.append(sub_array[1][*slices])
                        except:
                            print(sub_array)
            return ak.Array(new_array)


        def extract_awkward_array(array: ak.Array, slices: tuple):
            new_array = []
            for row in array.to_list():
                row_array = []
                for pixel in row:
                    if np.all(np.isnan(pixel)):
                        row_array.append([[np.nan for _ in slices]])
                    else:
                        row_array.append(np.array(pixel)[*slices])
                new_array.append(row_array)
            return new_array

        """return [[subarr[1][*slices] for subarr in row] for row in arr]"""

        # """""""""
        results_array = ak.Array(results)
        with open("fit4.pkl", "wb") as f:
            dill.dump(results_array, f)
        results_array = dilly
        chi2_array = Array2D(results_array[:,:,0].to_list())
        # print(results_array[:,:,1].to_list())
        # extract_awkward_array(results_array[:,:,1], (slice(None), slice(0, 2)))
        fit_array = ak.Array([
            extract_awkward_array(results_array[:,:,1], (slice(None), slice(i, 2+i))) for i in range(0, 6, 2)
        ]) 
        # """""""""






        # print(fit_array[0,:,:,:,0])
        amps_1 = fit_array[0,:,:,0,0].to_numpy()

        return Array2D(amps_1)


        # Convert back to Awkward Array and print the result



        #     fit_fwhm_map = np.array(pool.starmap(worker_split_fit, [(NII_data_cube(fits.PrimaryHDU(data[:,y,:], 
        #                                                                     self.header), double_NII_peak_authorized),
        #                                                             data.shape[1]) for y in range(data.shape[1])]))
        
        # new_header = self.header.flatten()
        # # The fit_fwhm_map has 5 dimensions (x_shape,y_shape,3,7,3) and the last three dimensions are given at every 
        # # pixel
        # # Third dimension: all three gaussian parameters. 0: fwhm, 1: amplitude, 2: mean
        # # Fourth dimension: all rays in the data_cube. 0: OH1, 1: OH2, 2: OH3, 3: OH4, 4: NII, 5: Ha, 6: 7 components 
        # # fit map
        # # Fifth dimension: 0: data, 1: uncertainties, 2: snr (only when associated to fwhm values)
        # # The 7 components fit map is a map taking the value 0 if a single component was fitted onto NII and the value 
        # # 1 iftwo components were considered
        # fwhm_maps = Maps([
        #     Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,0,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,0,0,1]),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,0,0,2])]), name="OH1_fwhm"),
        #     Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,1,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,0,1,1]),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,0,1,2])]), name="OH2_fwhm"),
        #     Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,2,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,0,2,1]),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,0,2,2])]), name="OH3_fwhm"),
        #     Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,3,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,0,3,1]),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,0,3,2])]), name="OH4_fwhm"),
        #     Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,4,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,0,4,1]),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,0,4,2])]), name="NII_fwhm"),
        #     Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,5,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,0,5,1]),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,0,5,2])]), name="Ha_fwhm"),
        #     Map(fits.PrimaryHDU(fit_fwhm_map[:,:,0,6,0], new_header), name="7_components_fit")
        # ])
        # amplitude_maps = Maps([
        #     Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,0,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,1,0,1])]), name="OH1_amplitude"),
        #     Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,1,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,1,1,1])]), name="OH2_amplitude"),
        #     Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,2,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,1,2,1])]), name="OH3_amplitude"),
        #     Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,3,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,1,3,1])]), name="OH4_amplitude"),
        #     Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,4,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,1,4,1])]), name="NII_amplitude"),
        #     Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,5,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,1,5,1])]), name="Ha_amplitude"),
        #     Map(fits.PrimaryHDU(fit_fwhm_map[:,:,1,6,0], new_header), name="7_components_fit")
        # ])
        # mean_maps = Maps([
        #     Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,0,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,2,0,1])]), name="OH1_mean"),
        #     Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,1,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,2,1,1])]), name="OH2_mean"),
        #     Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,2,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,2,2,1])]), name="OH3_mean"),
        #     Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,3,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,2,3,1])]), name="OH4_mean"),
        #     Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,4,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,2,4,1])]), name="NII_mean"),
        #     Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,5,0], new_header),
        #                         fits.ImageHDU(fit_fwhm_map[:,:,2,5,1])]), name="Ha_mean"),
        #     Map(fits.PrimaryHDU(fit_fwhm_map[:,:,2,6,0], new_header), name="7_components_fit")
        # ])
        # parameter_names = {"FWHM": fwhm_maps, "amplitude": amplitude_maps, "mean": mean_maps}
        # return_list = []
        # for element in extract:
        #     return_list.append(parameter_names[element])
        # if len(extract) == 1:
        #     # If only a single Maps is present, the element itself needs to be returned and not a list of a single 
        #     # element
        #     return return_list[-1]
        # else:
        #     return return_list
        
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
        """
        map_ = self[:,row,:]
        results = []
        nans = [np.NAN, [[np.NAN]]]
        for spectrum in map_:
            if np.all(np.isnan(spectrum.data)):
                # Empty spectrum
                results.append(nans)

            else:
                spectrum = spectrum.from_spectrum(SpectrumCO)
                if self.spectrum_parameters is not None:
                    for param, value in self.spectrum_parameters.items():
                        setattr(spectrum, param, value)

                spectrum.fit()

                if spectrum.is_successfully_fitted:
                    results.append([
                        spectrum.get_fit_chi2(),
                        spectrum.fit_results.to_numpy()
                    ])
                else:
                    results.append(nans)

        return results
