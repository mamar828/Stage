import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings

from astropy.io import fits
from scipy.interpolate import splrep, BSpline

from cube_spectrum import Spectrum

from datetime import datetime


if not sys.warnoptions:
    warnings.simplefilter("ignore")


class Data_cube_analyzer():

    def __init__(self, data_cube_file_name=str):
        self.data_cube = (fits.open(data_cube_file_name)[0]).data
        # self.data_cube = self.data_cube[:,:12,:6]
        # self.fit_equation = self.fit_spline(self.get_instrumental_widths())

    def fit_spline(self, array, s=150):
        # Remove [204f;215] and [402,415]
        array_sep = np.concatenate((array[:204,:], array[216:402,:], array[414:,:]))
        spl = splrep(array_sep[:,0], array_sep[:,1], s=s)

        x = np.linspace(1,484,1000)
        plt.plot(x, BSpline(*spl)(x), "g", linewidth=0.8)
        # plt.show()
        return spl
    
    def estimate_uncertainty(self):
        pass

    def fit_function(self, x):
        return BSpline(*self.fit_equation)(x)
    
    def fit_map(self):
        self.fit_fwhm_map = np.zeros([self.data_cube.shape[1], self.data_cube.shape[2], 2])
        for x in range(0, self.data_cube.shape[2]):
            if x%10 == 0:
                print("\n", x, end=" ")
            else:
                print(".", end="")
            for y in range(self.data_cube.shape[1]):
                # print(y)
                try:
                    spectrum_object = Spectrum(self.data_cube[:,y,x])
                    spectrum_object.fit()
                    self.fit_fwhm_map[y,x,:] = (spectrum_object.get_FWHM_speed(
                        spectrum_object.get_fitted_gaussian_parameters(), spectrum_object.get_uncertainties()["g0"]["stddev"]))
                except:
                    self.fit_fwhm_map[y,x,:] = [np.NAN, np.NAN]
        
        # In the matrix, every vertical group is a y coordinate, starting from (1,1) at the top
        # Every element in a group is a x coordinate
        # Every sub-element is the fwhm and its uncertainty
        # file = open("gaussian_fitting/fwhm_map.txt", "a")
        # file.write((str(datetime.now()) + "\n" + str(self.fit_fwhm_map) + "\n\n\n\n" + "".join(list("-" for _ in range(133))) + "\n\n\n\n"))
        self.save_as_fits_file("gaussian_fitting/instr_func1.fits", self.fit_fwhm_map[:,:,0])
        self.save_as_fits_file("gaussian_fitting/instr_func1_unc.fits", self.fit_fwhm_map[:,:,1])
        return self.fit_fwhm_map
    
    def save_as_fits_file(self, filename, array, header=None):
        fits.writeto(filename, array, header, overwrite=True)
    
    def plot_map(self, map):
        plt.colorbar(plt.imshow(map, origin="lower", cmap="viridis", vmin=15, vmax=50))
        plt.show()

    def bin_map(self, map):
        nb_pix_bin = 2
        for i in range(nb_pix_bin):
            try:
                bin_array = map.reshape(int(map.shape[0]/nb_pix_bin), nb_pix_bin, int(map.shape[1]/nb_pix_bin), nb_pix_bin)
                break
            except ValueError:
                map = np.resize(map, (map.shape[0]-1, map.shape[1]-1))
        new_values = bin_array.mean(axis=(1,3))
        return new_values

    def smooth_order_change(self, array, center=tuple):
        # Finds first the radiuses where a change of diffraction order can be seen
        center = round(center[0]), round(center[1])
        bin_factor = center[0] / 527
        smoothing_max_thresholds = [0.4, 1.8]
        bounds = [
            np.array((255,355)) * bin_factor,
            np.array((70,170)) * bin_factor
        ]
        regions = [
            list(array[center[1], int(bounds[0][0]):int(bounds[0][1])]),
            list(array[center[1], int(bounds[1][0]):int(bounds[1][1])])
        ]
        radiuses = [
            center[0] - (regions[0].index(min(regions[0])) + 255),
            center[0] - (regions[1].index(min(regions[1])) + 70)
        ]
        smooth_array = np.copy(array)
        for x in range(array.shape[1]):
            for y in range(array.shape[0]):
                current_radius = np.sqrt((x-center[0])**2 + (y-center[1])**2)
                if (radiuses[0] - 5*bin_factor <= current_radius <= radiuses[0] + 5*bin_factor or
                    radiuses[1] - 4*bin_factor <= current_radius <= radiuses[1] + 4*bin_factor):
                    mean_array = np.copy(array[y-3:y+4, x-3:x+4])
                    if radiuses[0] - 4*bin_factor <= current_radius <= radiuses[0] + 4*bin_factor:
                        mean_array[mean_array < np.max(mean_array)-smoothing_max_thresholds[0]] = np.NAN
                    else:
                        mean_array[mean_array < np.max(mean_array)-smoothing_max_thresholds[1]] = np.NAN
                    smooth_array[y,x] = np.nanmean(mean_array)
        return smooth_array

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
        plt.show()
        return np.array(list(zip(np.arange(484)+1, np.array(widths)[:,0])))
    
    def get_individual_instrumental_widths(self):
        widths_s = []
        for radius in range(1,485):
            widths_s.append(self.get_FWHM_mean(radius))
        # plt.plot(np.arange(484), np.array(widths_s)[:,0], label="total", alpha=0.7)
        widths = []
        for radius in range(1,485):
            widths.append(self.get_individual_FWHM_mean(radius))
        widths = np.array(widths)
        # plt.plot(np.arange(484), np.array(widths)[:,0,0], label="y-", alpha=0.6)
        # plt.plot(np.arange(484), np.array(widths)[:,1,0], label="y+", alpha=0.6)
        plt.plot(np.arange(484), np.array(widths)[:,2,0], label="x-", alpha=0.6)
        plt.plot(np.arange(484), np.array(widths)[:,3,0], label="x+", alpha=0.6)
        plt.legend(loc="upper left")
        plt.show()

    def get_corrected_width(self, spectrum=Spectrum):
        raw_fwhm = spectrum.get_FWHM_speed(spectrum.fit_iteratively()[4], spectrum.get_uncertainties()["g4"]["stddev"])
        return [raw_fwhm[0] - self.fit_function(200), raw_fwhm[1]]
        # return [raw_fwhm[0] - self.fit_function(200), raw_fwhm[1] + self.estimate_uncertainty()]



# file = fits.open("gaussian_fitting/instr_func.fits")[0].data
# header = fits.open("gaussian_fitting/instr_func.fits")[0].header



analyzer = Data_cube_analyzer("calibration.fits")
# analyzer.smooth_order_change(file, (527,484))
# analyzer.fit_map()
# analyzer.plot_map(analyzer.fit_fwhm_map[:,:,0])

fit_file = fits.open("maps/smoothed_instr_func.fits")[0].data
analyzer.plot_map(fit_file)

# analyzer.save_as_fits_file("maps/smoothed_instr_func.fits", analyzer.smooth_order_change(file, (527,484)))

# analyzer.plot_map(fit_file)
# analyzer.plot_map(analyzer.bin_map(fit_file))
# plt.colorbar(plt.imshow(fit_file, origin="lower", cmap="viridis", vmin=15, vmax=50))
# plt.show()

# sh = Data_cube_analyzer("gaussian_fitting/instr_func.fits")
# sh.bin_map(sh.data_cube)

# nuit_3 = fits.open("lambda_3.fits")[0].data
# nuit_4 = fits.open("lambda_4.fits")[0].data
# header = fits.open("lambda_3.fits")[0].header

# nuit_34 = np.flip(np.sum((nuit_3, nuit_4), axis=0), axis=(1,2))
# plt.imshow(nuit_34[15,:,:])
# plt.show()
# fits.writeto("night_34.fits", nuit_34, header, overwrite=True)
# print("d")

# hdu = fits.PrimaryHDU(nuit_34)
# hdu.writeto("night_34.fits", header, overwrite=True)


# plt.imshow(nuit_34[15,:,:])
# plt.show()


# analyzer.bin_map(analyzer.fit_fwhm_map[:,:,0])
# analyzer.get_center_point()
# print(analyzer.get_instrumental_widths())
# print(analyzer.get_individual_instrumental_widths())
# s = 0
# while True:
#     print(s)
#     analyzer.fit_spline(analyzer.get_instrumental_widths(),s)
#     s += 10
# analyzer.fit_spline(analyzer.get_instrumental_widths(),150)
# data = fits.open("cube_NII_Sh158_with_header.fits")[0].data
# spectrum_object = Spectrum(data[:,100-1,100-1], calibration=False)
# print(analyzer.get_corrected_width(spectrum_object))


# data_cube = np.flip(fits.open("calibration.fits")[0].data, axis=1)
# data_cube = fits.open("calibration.fits")[0].data
# data_cube_2 = fits.open("cube_NII_Sh158_with_header.fits")[0].data
# print(data_cube.shape)
# print(data_cube_2.shape)

# test = data_cube[35,:,:]
# masked = np.ma.masked_less(test, 1000)
# plt.imshow(masked, origin="lower")
# plt.plot(527,484,marker="o",color="white")
# plt.show()

# x, y = 500, 500

# print(data_cube[:,y-1,x-1])

# print(data_cube.shape)


