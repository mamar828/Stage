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
        x_mean, y_mean = np.mean(distances["x"], axis=(0,1)), np.mean(distances["y"], axis=(0,1))
        x_uncertainty, y_uncertainty = np.std(distances["x"], axis=(0,1)), np.std(distances["y"], axis=(0,1))
        return [x_mean, x_uncertainty], [y_mean, y_uncertainty]

    def get_FWHM_mean(self, px_radius):
        # Storage of the Fabry-Perot's center point on the image plot (+1 on DS9)
        image_center_x, image_center_y = 527, 484
        # Storage of the center point in the data array: the x and y coordinates are swapped
        center_x, center_y = image_center_y, image_center_x

        positions = [
            (center_x - px_radius, center_y),
            (center_x + px_radius, center_y),
            (center_x, center_y - px_radius),
            (center_x, center_y + px_radius)
        ]

        mean_y_values = (
            self.data_cube[:, center_x - px_radius, center_y] + self.data_cube[:, center_x + px_radius, center_y] +
            self.data_cube[:, center_x, center_y - px_radius] + self.data_cube[:, center_x, center_y + px_radius]
            ) / 4
        pixel = Spectrum(mean_y_values, calibration=True)
        pixel.fit()
        return pixel.get_FWHM_speed(pixel.fitted_gaussian, pixel.get_uncertainties()["g0"]["stddev"])

    def get_instrumental_width(self):
        widths = []
        for radius in range(1,485):
            widths.append(self.get_FWHM_mean(radius))
        arr = np.array(widths)
        plt.plot(np.arange(484), arr[:,0])
        plt.show()
        return widths
    
    def get_corrected_width(self):
        pass



analyzer = Calibration_cube_analyzer("calibration.fits")
# analyzer.get_center_point()
print(analyzer.get_instrumental_width())



# test = data_cube[35,:,:]
# masked = np.ma.masked_less(test, 610)
# plt.imshow(test, origin="lower")
# plt.plot(527,484,marker="o",color="white")
# plt.show()

x, y = 500, 500

# print(data_cube[:,y-1,x-1])

# print(data_cube.shape)


