This folder contains all the files specific to the analysis of the region Sh2-158.
To understand the code best, see the main.py file which gives examples for code usage.


arrays                  numpy arrays of the ACF and structure function of maps depending on values that were binned together.
data_cubes              .fits data cubes used for calculations.
leo                     .fits files taken from Leo's folder and slightly modified maps/data cubes.
maps                    computed maps.
        computed_data           maps obtained with the night_34_wcs.fits data cube with a single NII fit and without any
                                restrictions on the fit's quality.
        computed_data_2p        maps obtained with the night_34_wcs.fits data cube by allowing double NII fits and without any
                                restrictions on the fit's quality.
        computed_data_selective maps obtained with the night_34_wcs.fits data cube with a single NII fit and with additional
                                restrictions on the fit's quality.
        external_maps           maps obtained from external sources necessary for internal calculations
        new_leo                 maps obtained by fitting leo's cubes.
        reproject               maps with specific headers that each match a specific region on the object. These are used
                                to align maps on different regions
        SII                     maps obtained by fitting of the SII data cubes
        temp_maps_courtes       temperature maps obtained with the Courtes method with two different ions
other                   random test python files.
regions                 .reg regions representing the diffuse region, the central region and the filament region respectively.
results                 turbulence statistics, maps and histograms.
statistics              statistics of all three regions of different maps.
tests_large_regions     data, statistics and .reg files of the large spectrums used to compute temperatures in 
                        main.get_temp_NII_SII().
celestial_coords.py     classes specific to celestial coordinates and WCS adjustment.
cube_spectrum.py        classes specific to spectrum analysis of data cubes.
fits_analyzer.py        classes specific to analysis and calculations with data cubes and maps.
main.py                 commented examples of how the code is used to obtain results.
map_modifications.py    file used to modify the maps and adjust their WCS. This file is not explained but can be refered to for
                        understanding better how astropy.io.fits.Header objects can be modified.
zfilter.pro             raw Zurflueh filter code written in IDL.
zfilter.py              modified Zurflueh filter in python. This filter is not verified.
