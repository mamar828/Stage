import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from scipy.signal import argrelextrema

from cube_spectrum import Spectrum

# data_cube = np.flip(fits.open("calibration.fits")[0].data, axis=1)
data_cube = fits.open("calibration.fits")[0].data

test = data_cube[30,:,:]

# MIDDLE PIXEL (522,489)

# plt.imshow(data_cube[0,:,:])

class Calibration_cube_analyzer():

    def __init__(self, data_cube_file_name=str):
        self.data_cube = (fits.open(data_cube_file_name)[0]).data

    def find_center_point(self):
        center_guess = 511, 511
        distances = {"x": {}, "y": {}}
        for channel in range(48):
            print("channel:", channel)
            channel_pos = {}
            intensity_x_list = self.data_cube[channel, center_guess[1], :]
            intensity_y_list = self.data_cube[channel, :, center_guess[0]]
            for axes in range(2):
                last_intensity = 0
                axes_pos = []
                for coord in range(1,1024):
                    if intensity_x_list[coord-1] < intensity_x_list[coord] > intensity_x_list[coord+1] and intensity_x_list[coord] > 500:
                        axes_pos.append(coord - 1)
                    # new_intensity = intensity_x_list[coord]
                    # if new_intensity > 500 and new_intensity < last_intensity:
                    #     axes_pos.append(coord - 1)
                    # last_intensity = new_intensity
                print(axes_pos)
                raise ArithmeticError




        # channel = 0 
        # intensity_x_list = self.data_cube[channel, center_guess[1], :]
        # # Research of local maxes in the [10:300]
        # local_x_max_candidates = np.ma.masked_where(intensity_x_list < 500, intensity_x_list)
        # local_x_max_candidates = local_x_max_candidates[local_x_max_candidates.mask == False]
        # for element in local_x_max_candidates:
        #     print(list(intensity_x_list).index(element))
        # local_x_max = local_x_max_candidates[argrelextrema(local_x_max_candidates, np.greater)[0]]
        # print(pos_local_x_max)
        # print(local_x_max_candidates[local_x_max_candidates != --])
        # print(local_x_max)
        # local_x_max = argrelextrema(x_list, np.greater)[0]
        # local_x_max_i = x_list[local_x_max]

            # print(local_x_max_i)
        raise ArithmeticError

    def get_FWHM_mean(self, px_radius):
        # Storage of the Fabry-Perot's center point on the image
        image_center_x, image_center_y = 522, 489
        # Storage of the center point in the data array: the x and y coordinates are swapped
        center_x, center_y = image_center_y, image_center_x

        pixel = Spectrum(self.data_cube[:,0,0], calibration=True)
        pixel.fit()
        fwhm = pixel.get_FWHM_speed(pixel.fitted_gaussian, pixel.get_uncertainties()["g0"]["stddev"])
        pixel.plot_fit()
        print("\n", fwhm, max(pixel.data), "\n")
        
        raise ArithmeticError

        positions = [
            (center_x - px_radius, center_y),
            (center_x + px_radius, center_y),
            (center_x, center_y - px_radius),
            (center_x, center_y + px_radius)
        ]

        for coordinates in positions:
            pixel = Spectrum(self.data_cube[:,coordinates[0],coordinates[1]], calibration=True)
            pixel.fit()
            fwhm = pixel.get_FWHM_speed(pixel.fitted_gaussian, pixel.get_uncertainties()["g0"]["stddev"])
            print(fwhm, max(pixel.data))



analyzer = Calibration_cube_analyzer("calibration.fits")
analyzer.find_center_point()




# masked = np.ma.masked_less(test, 610)
# print(masked[489,522])
# plt.imshow(masked, origin="lower")
# plt.plot()
# plt.show()

x, y = 500, 500

# print(data_cube[:,y-1,x-1])

# print(data_cube.shape)


