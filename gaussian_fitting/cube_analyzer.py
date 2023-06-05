import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings

from astropy.io import fits

from cube_spectrum import Spectrum

import multiprocessing
import time

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class Data_cube_analyzer():

    def __init__(self, data_cube_file_name=str):
        self.data_cube = fits.open(data_cube_file_name)[0].data
        self.data_cube = self.data_cube[:,:10,:10]

    def fit_calibration(self, data):
        self.fit_fwhm_map = np.zeros([data.shape[1], data.shape[2], 2])
        for x in range(0, data.shape[2]):
            if x%10 == 0:
                print("\n", x, end=" ")
            else:
                print(".", end="")
            for y in range(data.shape[1]):
                try:
                    spectrum_object = Spectrum(data[:,y,x], calibration=True)
                    spectrum_object.fit()
                    self.fit_fwhm_map[y,x,:] = (spectrum_object.get_FWHM_speed(
                        spectrum_object.get_fitted_gaussian_parameters(), spectrum_object.get_uncertainties()["g0"]["stddev"]))
                except:
                    print(f"Exception encountered at ({x},{y})")
                    self.fit_fwhm_map[y,x,:] = [np.NAN, np.NAN]
        
        # In the matrix, every vertical group is a y coordinate, starting from (1,1) at the top
        # Every element in a group is a x coordinate
        # Every sub-element is the fwhm and its uncertainty
        self.save_as_fits_file("gaussian_fitting/instr_func1.fits", self.fit_fwhm_map[:,:,0])
        self.save_as_fits_file("gaussian_fitting/instr_func1_unc.fits", self.fit_fwhm_map[:,:,1])
        return self.fit_fwhm_map
    
    def fit_NII_cube(self, data):
        self.fit_fwhm_map = np.zeros([data.shape[1], data.shape[2], 2])
        for x in range(data.shape[2]):
            print(f"\n{x}", end=" ")
            for y in range(data.shape[1]):
                if y%10 == 0:
                    print(".", end="")
                spectrum_object = Spectrum(data[:,y,x], calibration=False)
                spectrum_object.fit(spectrum_object.get_initial_guesses())
                print(spectrum_object.get_FWHM_speed(
                        spectrum_object.get_fitted_gaussian_parameters()[4], spectrum_object.get_uncertainties()["g4"]["stddev"]))
                try:
                    self.fit_fwhm_map[y,x,:] = spectrum_object.get_FWHM_speed(
                        spectrum_object.get_fitted_gaussian_parameters()[4], spectrum_object.get_uncertainties()["g4"]["stddev"])
                except:
                    print(f"Exception encountered at ({x},{y})")
                    self.fit_fwhm_map[y,x,:] = [np.NAN, np.NAN]
        
        # In the matrix, every vertical group is a y coordinate, starting from (1,1) at the top
        # Every element in a group is a x coordinate
        # Every sub-element is the fwhm and its uncertainty
        self.save_as_fits_file("maps/fwhm_NII.fits", self.fit_fwhm_map[:,:,0])
        self.save_as_fits_file("maps/fwhm_NII_unc.fits", self.fit_fwhm_map[:,:,1])
        return self.fit_fwhm_map

    def fit_NII_cube_multiprocessively(self, data):
        fit_fwhm_map = np.zeros([data.shape[1], data.shape[2], 2])
        pool = Pool(processes=2)
        for x in range(data.shape[2]):
            print(f"\n{x}", end=" ")
            pool.map(worker_fit, list((x, i) for i in range(data.shape[1])))
        # In the matrix, every vertical group is a y coordinate, starting from (1,1) at the top
        # Every element in a group is a x coordinate
        # Every sub-element is the fwhm and its uncertainty
        self.save_as_fits_file("maps/fwhm_NII.fits", self.fit_fwhm_map[:,:,0])
        self.save_as_fits_file("maps/fwhm_NII_unc.fits", self.fit_fwhm_map[:,:,1])
        return self.fit_fwhm_map

    def save_as_fits_file(self, filename, array, header=None):
        fits.writeto(filename, array, header, overwrite=True)
    
    def plot_map(self, map, autoscale=True, bounds=None):
        if autoscale:
            plt.colorbar(plt.imshow(map, origin="lower", cmap="viridis"))
        elif bounds:
            plt.colorbar(plt.imshow(map, origin="lower", cmap="viridis", vmin=bounds[0], vmax=bounds[1]))
        else:
            plt.colorbar(plt.imshow(map, origin="lower", cmap="viridis", vmin=map[round(map.shape[0]/2), round(map.shape[1]/2)]*3/5,
                                                                     vmax=map[round(map.shape[0]/10), round(map.shape[1]/10)]*2))
        plt.show()

    def bin_map(self, map, nb_pix_bin=2):
        for i in range(nb_pix_bin):
            try:
                bin_array = map.reshape(int(map.shape[0]/nb_pix_bin), nb_pix_bin, int(map.shape[1]/nb_pix_bin), nb_pix_bin)
                break
            except ValueError:
                print(f"Map to bin will be cut by {i+1} pixel(s).")
                map = map[:-1,:-1]
        return np.nanmean(bin_array, axis=(1,3))
    
    def bin_cube(self, cube, nb_pix_bin=2):
        for i in range(nb_pix_bin):
            try:
                bin_array = cube.reshape(cube.shape[0], int(cube.shape[1]/nb_pix_bin), nb_pix_bin,
                                                        int(cube.shape[2]/nb_pix_bin), nb_pix_bin)
                break
            except ValueError:
                print(f"Cube to bin will be cut by {i+1} pixel(s).")
                cube = cube[:,:-1,:-1]
        return np.nanmean(bin_array, axis=(2,4))

    def smooth_order_change(self, data_array, uncertainty_array, center=tuple):
        # Find first the radiuses where a change of diffraction order can be seen
        center = round(center[0]), round(center[1])
        bin_factor = center[0] / 527
        smoothing_max_thresholds = [0.4, 1.8]
        bounds = [
            np.array((255,355)) * bin_factor,
            np.array((70,170)) * bin_factor
        ]
        peak_regions = [
            list(data_array[center[1], int(bounds[0][0]):int(bounds[0][1])]),
            list(data_array[center[1], int(bounds[1][0]):int(bounds[1][1])])
        ]
        radiuses = [
            center[0] - (peak_regions[0].index(min(peak_regions[0])) + 255),
            center[0] - (peak_regions[1].index(min(peak_regions[1])) + 70)
        ]
        smooth_data = np.copy(data_array)
        smooth_uncertainties = np.copy(uncertainty_array)
        for x in range(data_array.shape[1]):
            for y in range(data_array.shape[0]):
                current_radius = np.sqrt((x-center[0])**2 + (y-center[1])**2)
                if (radiuses[0] - 5*bin_factor <= current_radius <= radiuses[0] + 5*bin_factor or
                    radiuses[1] - 4*bin_factor <= current_radius <= radiuses[1] + 4*bin_factor):
                    near_pixels = np.copy(data_array[y-3:y+4, x-3:x+4])
                    near_pixels_uncertainty = np.copy(uncertainty_array[y-3:y+4, x-3:x+4])
                    if radiuses[0] - 4*bin_factor <= current_radius <= radiuses[0] + 4*bin_factor:
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[0]] = np.NAN
                    else:
                        near_pixels[near_pixels < np.max(near_pixels)-smoothing_max_thresholds[1]] = np.NAN
                    smooth_data[y,x] = np.nanmean(near_pixels)
                    smooth_uncertainties[y,x] = np.nanmean(near_pixels * 0 + near_pixels_uncertainty)
        return smooth_data, smooth_uncertainties

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

    def get_corrected_width(self, fwhm_NII, fwhm_NII_uncertainty,
                            instrumental_function_width, instrumental_function_width_uncertainty):
        return [fwhm_NII - instrumental_function_width,
                fwhm_NII_uncertainty + instrumental_function_width_uncertainty]
    

fit_state = np.array(list(list(0 for _ in range(51)) for _ in range(51)))
np.set_printoptions(threshold=sys.maxsize)

def update_fit_state():
    print(np.array_str(fit_state))

def worker_fit(args):
    y, data = args
    line = []
    for x in range(data.shape[1]):
        if y%10 == 0 and x%10 == 0:
            fit_state[y,x] = 1
            update_fit_state()
        spectrum_object = Spectrum(data[:,y,x], calibration=False)
        spectrum_object.fit(spectrum_object.get_initial_guesses())
        line.append(spectrum_object.get_FWHM_speed(
                    spectrum_object.get_fitted_gaussian_parameters()[4], spectrum_object.get_uncertainties()["g4"]["stddev"]))
        if y%10 == 0 and x%10 == 0:
            fit_state[y,x] = 2
            update_fit_state()
    return line

if __name__ == "__main__":
    analyzer = Data_cube_analyzer("night_34.fits")
    data = analyzer.bin_cube(analyzer.data_cube, 2)
    fit_fwhm_list = []
    update_fit_state()
    pool = multiprocessing.Pool()
    start = time.time()
    fit_fwhm_list.append(np.array(pool.map(worker_fit, list((y, data) for y in range(data.shape[1])))))
    stop = time.time()
    print(stop-start, "s")
    pool.close()
    fitted_array = np.squeeze(np.array(fit_fwhm_list), axis=0)
    # print(fitted_array)
    # analyzer.save_as_fits_file("maps/fwhm_NII.fits", fitted_array[:,:,0])
    # analyzer.save_as_fits_file("maps/fwhm_NII_unc.fits", fitted_array[:,:,1])




# file = fits.open("cube_NII_Sh158_with_header.fits")[0].data
fwhms = fits.open("maps/fwhm_NII.fits")[0].data
fwhms_unc = fits.open("maps/fwhm_NII_unc.fits")[0].data
calibs = fits.open("maps/smoothed_instr_f.fits")[0].data
calibs_unc = fits.open("maps/smoothed_instr_f_unc.fits")[0].data
corrected_fwhm = fits.open("maps/corrected_fwhm.fits")[0].data
corrected_fwhm_unc = fits.open("maps/corrected_fwhm_unc.fits")[0].data

# header = fits.open("night_34.fits")[0].header
# print(header)

analyzer = Data_cube_analyzer("night_34.fits")
# analyzer.fit_NII_cube(analyzer.bin_cube(analyzer.data_cube[:,:4,:4], 2))
# analyzer.plot_map(corrected_fwhm_unc, autoscale=False, bounds=(0,60))

# analyzer.plot_map(corrected_fwhm_unc, autoscale=False, bounds=(0,60))



# sp = Spectrum(analyzer.bin_cube(analyzer.data_cube)[:,223,309], calibration=False)
# sp.fit(sp.get_initial_guesses())
# sp.plot_fit()




# corrected_map = analyzer.get_corrected_width(fwhms, fwhms_unc, analyzer.bin_map(calibs), analyzer.bin_map(calibs_unc))
# analyzer.save_as_fits_file("maps/corrected_fwhm.fits", corrected_map[0])
# analyzer.save_as_fits_file("maps/corrected_fwhm_unc.fits", corrected_map[1])

