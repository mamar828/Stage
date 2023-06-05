if __name__ == "__main__":
    from cube_analyzer import Data_cube_analyzer
    from cube_spectrum import Spectrum

    from multiprocessing import Pool

    import numpy as np        


    analyzer = Data_cube_analyzer("night_34.fits")
    data = analyzer.bin_cube(analyzer.data_cube, 2)
    fit_fwhm_map = np.zeros([data.shape[1], data.shape[2], 2])
    pool = Pool(processes=2)
    for x in range(data.shape[2]):
        print(f"\n{x}", end=" ")
        pool.map(worker_fit, list((x, i) for i in range(data.shape[1])))
    # In the matrix, every vertical group is a y coordinate, starting from (1,1) at the top
    # Every element in a group is a x coordinate
    # Every sub-element is the fwhm and its uncertainty
    # self.save_as_fits_file("maps/fwhm_NII.fits", self.fit_fwhm_map[:,:,0])
    # self.save_as_fits_file("maps/fwhm_NII_unc.fits", self.fit_fwhm_map[:,:,1])
