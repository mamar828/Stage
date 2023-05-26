from astropy.io import fits

from cube_spectrum import Spectrum

data_cube = fits.open("calibration.fits")[0].data

x, y = 500, 500
print(data_cube[:,y-1,x-1])

print(data_cube.shape)


