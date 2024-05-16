import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from eztcolors import Colors as C

from src.hdu.cubes.cube import Cube


class CubeCO(Cube):
    """
    Encapsulates the methods specific to CO data cubes.
    """

    def fit(self, extract: list[str], double_NII_peak_authorized: bool=False) -> list[Maps]:
        """
        Fit the whole data cube to extract the peaks' data. This method presupposes that four OH peaks, one Halpha 
        peak and one NII peak (sometimes two if the seven_components_fits_authorized is set to True) are present.
        WARNING: Due to the use of the multiprocessing library, calls to this function NEED to be made inside a 
        condition state with the following phrasing:
        if __name__ == "__main__":
        This prevents the code to recursively create instances of itself that would eventually overload the CPUs.

        Arguments
        ---------
        extract: list of str. Names of the gaussians' parameters to extract. Supported terms are: "mean", "amplitude" 
        and "FWHM". Any combination or number of these terms can be given.
        double_NII_peak_authorized: bool, default=False. Specifies if double NII peaks are considered possible. If 
        True, the fitting algorithm may detect two components and fit these separately. The NII values in the returned 
        maps for a certain parameter will then be the mean value of the two fitted gaussians at that pixel.

        Returns
        -------
        list of Maps object: the Maps object representing every ray's mean, amplitude or FWHM are returned in the 
        order they were put in argument, thus the list may have a length of 1, 2 or 3. Every Maps object has the maps 
        of every ray present in the provided data cube.
        Note that each map is a Map_usnr object when computing the FWHM whereas the maps are Map_u objects when 
        computing amplitude or mean. In any case, in every Maps object is a Map object having the value 1 when a seven 
        components fit was executed and 0 otherwise.
        """
        self.verify_extract_list(extract)

        data = np.copy(self.data)
        with Pool() as pool:
            print(f"{C.YELLOW}Number of processes used: {pool._processes}{C.END}")
            fit_fwhm_map = np.array(pool.starmap(worker_split_fit, [(NII_data_cube(fits.PrimaryHDU(data[:,y,:], 
                                                                            self.header), double_NII_peak_authorized),
                                                                    data.shape[1]) for y in range(data.shape[1])]))
        
        new_header = self.header.get_flattened()
        # The fit_fwhm_map has 5 dimensions (x_shape,y_shape,3,7,3) and the last three dimensions are given at every 
        # pixel
        # Third dimension: all three gaussian parameters. 0: fwhm, 1: amplitude, 2: mean
        # Fourth dimension: all rays in the data_cube. 0: OH1, 1: OH2, 2: OH3, 3: OH4, 4: NII, 5: Ha, 6: 7 components 
        # fit map
        # Fifth dimension: 0: data, 1: uncertainties, 2: snr (only when associated to fwhm values)
        # The 7 components fit map is a map taking the value 0 if a single component was fitted onto NII and the value 
        # 1 iftwo components were considered
        fwhm_maps = Maps([
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,0,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,0,0,1]),
                                fits.ImageHDU(fit_fwhm_map[:,:,0,0,2])]), name="OH1_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,1,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,0,1,1]),
                                fits.ImageHDU(fit_fwhm_map[:,:,0,1,2])]), name="OH2_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,2,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,0,2,1]),
                                fits.ImageHDU(fit_fwhm_map[:,:,0,2,2])]), name="OH3_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,3,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,0,3,1]),
                                fits.ImageHDU(fit_fwhm_map[:,:,0,3,2])]), name="OH4_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,4,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,0,4,1]),
                                fits.ImageHDU(fit_fwhm_map[:,:,0,4,2])]), name="NII_fwhm"),
            Map_usnr(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,0,5,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,0,5,1]),
                                fits.ImageHDU(fit_fwhm_map[:,:,0,5,2])]), name="Ha_fwhm"),
            Map(fits.PrimaryHDU(fit_fwhm_map[:,:,0,6,0], new_header), name="7_components_fit")
        ])
        amplitude_maps = Maps([
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,0,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,0,1])]), name="OH1_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,1,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,1,1])]), name="OH2_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,2,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,2,1])]), name="OH3_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,3,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,3,1])]), name="OH4_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,4,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,4,1])]), name="NII_amplitude"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,1,5,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,1,5,1])]), name="Ha_amplitude"),
            Map(fits.PrimaryHDU(fit_fwhm_map[:,:,1,6,0], new_header), name="7_components_fit")
        ])
        mean_maps = Maps([
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,0,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,0,1])]), name="OH1_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,1,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,1,1])]), name="OH2_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,2,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,2,1])]), name="OH3_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,3,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,3,1])]), name="OH4_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,4,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,4,1])]), name="NII_mean"),
            Map_u(fits.HDUList([fits.PrimaryHDU(fit_fwhm_map[:,:,2,5,0], new_header),
                                fits.ImageHDU(fit_fwhm_map[:,:,2,5,1])]), name="Ha_mean"),
            Map(fits.PrimaryHDU(fit_fwhm_map[:,:,2,6,0], new_header), name="7_components_fit")
        ])
        parameter_names = {"FWHM": fwhm_maps, "amplitude": amplitude_maps, "mean": mean_maps}
        return_list = []
        for element in extract:
            return_list.append(parameter_names[element])
        if len(extract) == 1:
            # If only a single Maps is present, the element itself needs to be returned and not a list of a single 
            # element
            return return_list[-1]
        else:
            return return_list
        
    def worker_fit(self, x: int) -> list:
        """
        Fit a pixel of a 2 dimensional NII_data_cube, i.e., a Data_cube that is a simple line.

        Arguments
        ---------
        x: int. x coordinate of the pixel that needs to be fitted

        Returns
        -------
        list: FWHM, snr, amplitude and mean value of every fitted gaussians are given along with a map representing if 
        a double NII fit was made. Each coordinate has three sublists. The first has seven values: the first six are 
        the peaks' FWHM with their uncertainty and signal to noise ratio and the last one is a map indicating where  
        fits with seven components were done. The last map outputs 0 for a six components fit and 1 for a seven 
        components fit. The second sublist gives the fitted gaussians amplitude and the third sublist gives their mean 
        value. The two last sublists also have the map that represents if a double NII peak was fitted.
        """
        spectrum = NII_spectrum(self.data[:,x], self.header, 
                                seven_components_fit_authorized=self._double_NII_peak_authorized)
        spectrum.fit()
        if spectrum.is_nicely_fitted_for_NII():
            return [
                spectrum.get_FWHM_snr_7_components_array(), 
                spectrum.get_amplitude_7_components_array(), 
                spectrum.get_mean_7_components_array()
            ]
        else:
            return spectrum.get_list_of_NaN_arrays()
