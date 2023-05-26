import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

from cube_spectrum import Spectrum

# data_cube = np.flip(fits.open("calibration.fits")[0].data, axis=1)
data_cube = fits.open("calibration.fits")[0].data

test = data_cube[30,:,:]

# MIDDLE PIXEL (522,489)

# plt.imshow(data_cube[0,:,:])

class Calibration_cube_analyzer():

    def __init__(self, data_cube_file_name=str):
        self.data_cube = np.flip(fits.open(data_cube_file_name)[0].data, axis=1)

    def get_FWHM_mean(self, px_radius):
        center_x, center_y = 534, 522
        iterables = [
            (center_x - px_radius, center_y),
            (center_x + px_radius, center_y),
            (center_x, center_y - px_radius),
            (center_x, center_y + px_radius)
        ]

        for coordinates in iterables:
            pixel = Spectrum(self.data_cube[:,coordinates[0],coordinates[1]], calibration=True)
            pixel.fit()
            fwhm = pixel.get_FWHM_speed(pixel.fitted_gaussian, pixel.get_uncertainties()["g0"]["stddev"])
            print(fwhm)



analyzer = Calibration_cube_analyzer("calibration.fits")
analyzer.get_FWHM_mean(1)




masked = np.ma.masked_less(test, 610)
print(masked[534,522])
plt.imshow(masked, origin="lower")
plt.plot()
plt.show()

x, y = 500, 500

# print(data_cube[:,y-1,x-1])

# print(data_cube.shape)


