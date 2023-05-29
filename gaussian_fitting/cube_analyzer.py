import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from scipy.signal import argrelextrema

from cube_spectrum import Spectrum

# data_cube = np.flip(fits.open("calibration.fits")[0].data, axis=1)
data_cube = fits.open("calibration.fits")[0].data

# MIDDLE PIXEL (522,489)

# plt.imshow(data_cube[0,:,:])

class Calibration_cube_analyzer():

    def __init__(self, data_cube_file_name=str):
        self.data_cube = (fits.open(data_cube_file_name)[0]).data

    def get_center_point(self):
        center_guess = 527, 484
        distances = {"x": [], "y": []}
        success = 0
        for channel in range(1,49):
            print("channel:", channel)
            channel_dist = {}
            intensity_max = {
                "intensity_max_x": self.data_cube[channel-1, center_guess[1], :],
                "intensity_max_y": self.data_cube[channel-1, :, center_guess[0]]
            }
            for name, axe_list in intensity_max.items():
                axes_pos = []
                for coord in range(1, 1023):
                    if (axe_list[coord-1] < axe_list[coord] > axe_list[coord+1]
                         and axe_list[coord] > 500):
                        axes_pos.append(coord)
                # Verification of single maxes
                for i in range(len(axes_pos)-1):
                    if axes_pos[i+1] - axes_pos[i] < 50:
                        if axe_list[axes_pos[i]] > axe_list[axes_pos[i+1]]:
                            axes_pos[i+1] = 0
                        else:
                            axes_pos[i] = 0
                axes_pos = np.setdiff1d(axes_pos, 0)
                
                if len(axes_pos) == 4:
                    channel_dist[name] = [(axes_pos[3] - axes_pos[0])/2 + axes_pos[0], (axes_pos[2] - axes_pos[1])/2 + axes_pos[1]]
                    distances[name[-1]].append(channel_dist[name])
                    success += 1
                else:
                    print(f"channel {channel} not len == 4, len == {len(axes_pos)}, {axes_pos}")
            # print(distances)
        x_mean, y_mean = np.mean(distances["x"], axis=(0,1)), np.mean(distances["y"], axis=(0,1))
        print(x_mean, y_mean)
        print("success=", success)
        return x_mean, y_mean




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
        image_center_x, image_center_y = 527, 483
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
analyzer.get_center_point()
# analyzer.get_FWHM_mean(5)



test = data_cube[35,:,:]
masked = np.ma.masked_less(test, 610)
print(masked[489,522])
plt.imshow(test, origin="lower")
plt.plot(527,484,marker="v",color="white")
plt.show()

x, y = 500, 500

# print(data_cube[:,y-1,x-1])

# print(data_cube.shape)


