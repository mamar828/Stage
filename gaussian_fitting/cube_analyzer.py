import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from scipy.interpolate import splrep, BSpline

from cube_spectrum import Spectrum



class Calibration_cube_analyzer():

    def __init__(self, data_cube_file_name=str):
        self.data_cube = (fits.open(data_cube_file_name)[0]).data
        self.fit_equation = self.fit_spline(self.get_instrumental_widths())

    def fit_spline(self, array, s=150):
        # Remove [204;215] and [402,415]
        array_sep = np.concatenate((array[:204,:], array[216:402,:], array[414:,:]))
        spl = splrep(array_sep[:,0], array_sep[:,1], s=s)

        x = np.linspace(1,484,1000)
        plt.plot(x, BSpline(*spl)(x), "g", linewidth=0.8)
        # plt.show()
        return spl
    
    def fit_function(self, x):
        return BSpline(*self.fit_equation)(x)

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
        mean_y_values = (
            self.data_cube[:, center_x - px_radius, center_y] + self.data_cube[:, center_x + px_radius, center_y] +
            self.data_cube[:, center_x, center_y - px_radius] + self.data_cube[:, center_x, center_y + px_radius]
            ) / 4
        pixel = Spectrum(mean_y_values, calibration=True)
        pixel.fit()
        return pixel.get_FWHM_speed(pixel.fitted_gaussian, pixel.get_uncertainties()["g0"]["stddev"])

    def get_individual_FWHM_mean(self, px_radius):
        # Storage of the Fabry-Perot's center point on the image plot (+1 on DS9)
        image_center_x, image_center_y = 527, 484
        # Storage of the center point in the data array: the x and y coordinates are swapped
        center_x, center_y = image_center_y, image_center_x
        mean_y_values = (
            self.data_cube[:, center_x - px_radius, center_y] + self.data_cube[:, center_x + px_radius, center_y] +
            self.data_cube[:, center_x, center_y - px_radius] + self.data_cube[:, center_x, center_y + px_radius]
            ) / 4
        pixel1 = Spectrum(self.data_cube[:, center_x - px_radius, center_y], calibration=True)
        pixel2 = Spectrum(self.data_cube[:, center_x + px_radius, center_y], calibration=True)
        pixel3 = Spectrum(self.data_cube[:, center_x, center_y - px_radius], calibration=True)
        pixel4 = Spectrum(self.data_cube[:, center_x, center_y + px_radius], calibration=True)
        pixel1.fit()
        pixel2.fit()
        pixel3.fit()
        pixel4.fit()
        return (pixel1.get_FWHM_speed(pixel1.fitted_gaussian, pixel1.get_uncertainties()["g0"]["stddev"]),
                pixel2.get_FWHM_speed(pixel2.fitted_gaussian, pixel2.get_uncertainties()["g0"]["stddev"]),
                pixel3.get_FWHM_speed(pixel3.fitted_gaussian, pixel3.get_uncertainties()["g0"]["stddev"]),
                pixel4.get_FWHM_speed(pixel4.fitted_gaussian, pixel4.get_uncertainties()["g0"]["stddev"]))

    def get_instrumental_widths(self):
        widths = []
        for radius in range(1,485):
            widths.append(self.get_FWHM_mean(radius))
        plt.plot(np.arange(484), np.array(widths)[:,0], "r", linewidth=0.8)
        # plt.show()
        return np.array(list(zip(np.arange(484)+1, np.array(widths)[:,0])))
    
    def get_individual_instrumental_widths(self):
        widths_s = []
        for radius in range(1,485):
            widths_s.append(self.get_FWHM_mean(radius))
        plt.plot(np.arange(484), np.array(widths_s)[:,0], label="total", alpha=0.7)
        widths = []
        for radius in range(1,485):
            widths.append(self.get_individual_FWHM_mean(radius))
        widths = np.array(widths)
        plt.plot(np.arange(484), np.array(widths)[:,0,0], label="y-", alpha=0.6)
        plt.plot(np.arange(484), np.array(widths)[:,1,0], label="y+", alpha=0.6)
        plt.plot(np.arange(484), np.array(widths)[:,2,0], label="x-", alpha=0.6)
        plt.plot(np.arange(484), np.array(widths)[:,3,0], label="x+", alpha=0.6)
        plt.legend(loc="upper left")
        plt.show()
        return np.array(list(zip(np.arange(484)+1, np.array(widths_s)[:,0])))

    def get_corrected_width(self, spectrum=Spectrum):
        raw_fwhm = spectrum.get_FWHM_speed(spectrum.fit_iteratively()[4], spectrum.get_uncertainties()["g4"]["stddev"])
        return [raw_fwhm[0] - self.fit_function(200), raw_fwhm[1] ]



analyzer = Calibration_cube_analyzer("calibration.fits")
# analyzer.get_center_point()
# print(analyzer.get_instrumental_widths())
# print(analyzer.get_individual_instrumental_widths())
# s = 0
# while True:
#     print(s)
#     analyzer.fit_spline(analyzer.get_instrumental_widths(),s)
#     s += 10
# analyzer.fit_spline(analyzer.get_instrumental_widths(),150)
data = fits.open("cube_NII_Sh158_with_header.fits")[0].data
spectrum_object = Spectrum(data[:,100-1,100-1], calibration=False)
print(analyzer.get_corrected_width(spectrum_object))


# data_cube = np.flip(fits.open("calibration.fits")[0].data, axis=1)
# data_cube = fits.open("calibration.fits")[0].data


# test = data_cube[35,:,:]
# masked = np.ma.masked_less(test, 610)
# plt.imshow(test, origin="lower")
# plt.plot(527,484,marker="o",color="white")
# plt.show()

x, y = 500, 500

# print(data_cube[:,y-1,x-1])

# print(data_cube.shape)


