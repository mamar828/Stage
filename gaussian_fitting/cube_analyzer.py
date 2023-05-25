from astropy.io import fits

from cube_spectrum import Spectrum

data_cube = fits.open("calibration.fits")[0].data
print(data_cube.shape)