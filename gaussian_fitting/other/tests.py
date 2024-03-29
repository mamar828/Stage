import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
import os
from astropy.io import fits
import scipy
import uncertainties


"""x = np.array([x for x in range(101)])
y = np.e ** (- (0.1 * (x - 50)) ** 2)
y += np.array(list((np.random.random() - 0.5) * 0.25 for y in range(len(y))))




# Generate fake data
# rng = np.random.default_rng(0)
# x = np.linspace(-5., 5., 200)
# y = 3 * np.exp(-0.5 * (x - 1.3)**2 / 0.8**2)
# y += rng.normal(0., 0.2, x.shape)

# Fit the data using a Gaussian
g_init = models.Gaussian1D(amplitude=1., mean=50, stddev=1.)
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init, x, y)

# Plot the data with the best-fit model
plt.figure(figsize=(8,5))
plt.plot(x, y, 'ko')
plt.plot(x, g(x), label='Gaussian')
plt.xlabel('Position')
plt.ylabel('Flux')
plt.legend(loc=2)
plt.show()"""



# ----------------------------------------------------------------------------------------------------------



# data = np.fromfile(os.path.abspath("test.dat"), sep=" ")
# print(data)
# real_data = np.array(np.split(data, 2))
# print(real_data)


# fits_data = (fits.open(os.path.abspath("calibration.fits"))[0].data)
# fits_wcs = WCS((fits.open('calibration.fits', mode = 'denywrite'))[0].header)
# plt.plot(fits_data[:,150,150])
# plt.show()



# ----------------------------------------------------------------------------------------------------------


"""
data = (fits.open(os.path.abspath("calibration.fits"))[0].data)
rdata = data[:,0,600]

x_values, y_values = np.arange(48) + 1, rdata

g_init = models.Gaussian1D(amplitude=100., mean=30, stddev=1.)
fit_g = fitting.LevMarLSQFitter()
bounds = (26,32)

fitted_gaussian = fit_g(g_init, x_values[bounds[0]-1:bounds[1]], y_values[bounds[0]-1:bounds[1]])
mean = (np.sum(y_values[0:bounds[0]-1]) + np.sum(y_values[bounds[1]+1:48])) / (
    max(x_values) - (bounds[1] - bounds[0] + 1))



plt.plot(x_values, fitted_gaussian(x_values) + mean, label="fitted")
plt.plot(x_values, y_values, "y--", label="ds9 spectrum")
plt.legend(loc="upper left")
plt.xlabel("channels")
plt.ylabel("intensity")
plt.show()
"""



# --------------------------------------------------------------------


# a = True
# b = True
# c = False
# print(a == b and (a == a))


# -------------------------------------------------------------

# g_init = models.Voigt1D(x_0=self.max_tuple[0], amplitude_L=500, fwhm_L=3., fwhm_G=3.5)
        # g_init = models.Gaussian1D(amplitude=1., mean=self.max_tuple[0], stddev=1.)
        # g_init = gauss_function(self.x_values, a=500., x0=self.max_tuple, sigma=1., h=100.)
        # fit_g = fitting.LevMarLSQFitter()
        # bounds = self.get_peak_bounds()
        # self.fitted_gaussian = fit_g(g_init, self.x_values[bounds[0]-1:bounds[1]], self.y_values[bounds[0]-1:bounds[1]])



# -----------------------------------------------------------------------------

# stack = np.stack((x_values, self.fitted_gaussian(x_values)), axis=1)
# b = []
# for i in stack:
#     b.append(tuple(i))

# print(b)




# -------------------------------------------------------------------------------------------

# a = np.zeros(shape=(1,2))
# print(a)


# -------------------------------------------------------------------------------------------------------

# a = " [1.55726605e+01 2.88057290e+01 9.92512489e+00 4.79773984e+07"
# n = 0
# for i in a:
#     n += 1
# print(n)


# -----------------------------------------------------------------------------

# def func(a, **other):
#     print(a)
#     for e, i in other.items():
#         print(i)

# func("lol", lol=["binou", "toupie"])



# -----------------------------------------------------------------------------------


# numbers = [2, 3, 7, 4, 8]

# sum(number**2 for number in numbers if number % 2 == 0)
# exec("result = sum(number**2 for number in numbers if number % 2 == 0)")
# print(result)

# pro = "allo = 3"
# exec(pro)
# print(allo)




# -----------------------------------------

# a = int()

# print(a == 0)



# ---------------------------------------------

# fits_data = (fits.open('calibration.fits'))[0].data

# plt.plot(fits_data[:,0,0])
# plt.show()

# ------------------------------------------------------

# binou = None
# binou = 3
# if binou:
#     print(binou)

# ----------------------------------------------------------

# file = open("writer.txt", "a")
# file.write("SCHLOUBABOOBAfasddsfa\n")

# file = open("writer.txt", "r")
# print(file.read())


# --------------------------------------------------------------------

# nuit_3 = fits.open("lambda_3.fits")[0].data
# nuit_4 = fits.open("lambda_4.fits")[0].data
# header = fits.open("lambda_3.fits")[0].header

# nuit_34 = np.flip(np.sum((nuit_3, nuit_4), axis=0), axis=(1,2))
# plt.imshow(nuit_34[15,:,:])
# plt.show()
# # fits.writeto("night_34.fits", nuit_34, header, overwrite=True)
# print("d")

# ------------------------------------------------------------------------------------------------------------------------------

# import time, sys

# lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."

# for x in lorem:
#     print(x, end="")
#     time.sleep(0.03)
#     sys.stdout.flush()

# ------------------------------------------------------------------------------------

# a = np.array([[1,2], [3,4]])
# b = np.array([[5,6], [7,8]])
# print(np.stack((a,b), axis=2))

# -------------------------------------------------------

# from reproject import reproject_interp
# from astropy.wcs import WCS

# def Align(map_to_align, ref_map, name_of_new_map):
	
# 	"""
# 	Fonction réaligner une carte avec un mauvais WCS ou un décalage à partir d'une carte de référence avec un bon WCS.
# 	map_to_align: Path de la carte à réaligner. [String]
# 	ref_map: Path de la carte de référence. [String]
# 	name_of_new_map: Path de la carte réalignée. [String]
# 	Aller lire la documentation de la librairie Reporoject.
# 	"""
	
# 	#Ouverure de la carte à réaligner.
# 	hdu1 = fits.open(map_to_align, mode = 'denywrite')[0]
# 	#Ouverure de la carte de référence avec le bon WCS.
# 	hdu2 = fits.open(ref_map, mode = 'denywrite')[0]
# 	#Affichage de la carte à réaligner avec son WCS.	
# 	ax1 = plt.subplot(1,2,1, projection=WCS(hdu1.header))
# 	ax1.imshow(hdu1.data, origin='lower')
# 	ax1.coords.grid(color='white')
# 	ax1.coords['ra'].set_axislabel('Right Ascension')
# 	ax1.coords['dec'].set_axislabel('Declination')
# 	#ax1.set_title('Sh2-158 5755')
# 	#Affichage de la carte de référence avec son WCS.
# 	ax2 = plt.subplot(1,2,2, projection=WCS(hdu2.header))
# 	ax2.imshow(hdu2.data, origin='lower')
# 	ax2.coords.grid(color='white')
# 	ax2.coords['ra'].set_axislabel('Right Ascension')
# 	ax2.coords['dec'].set_axislabel('Declination')
# 	ax2.coords['dec'].set_axislabel_position('r')
# 	ax2.coords['dec'].set_ticklabel_position('r')
# 	#ax2.set_title('Sh2-158 6583')
# 	plt.show()
# 	#Utilisation de la librairie Reproject pour réaligner la carte déffecteuse sur celle de référence.
# 	array, footprint = reproject_interp(hdu1, hdu2.header)
# 	#Affichage de la carte réalignée avec son WCS.
# 	ax1 = plt.subplot(1,2,1, projection=WCS(hdu2.header))
# 	ax1.imshow(array, origin='lower')
# 	ax1.coords.grid(color='white')
# 	ax1.coords['ra'].set_axislabel('Right Ascension')
# 	ax1.coords['dec'].set_axislabel('Declination')
# 	#ax1.set_title('Reprojected Sh2-158 5755')
# 	#Affichage de la trace de la carte réalignée sur l'ancienne carte avec son WCS.
# 	ax2 = plt.subplot(1,2,2, projection=WCS(hdu2.header))
# 	ax2.imshow(footprint, origin='lower')
# 	ax2.coords.grid(color='white')
# 	ax1.coords['ra'].set_axislabel('Right Ascension')
# 	ax1.coords['dec'].set_axislabel('Declination')
# 	ax2.coords['dec'].set_axislabel_position('r')
# 	ax2.coords['dec'].set_ticklabel_position('r')
# 	#ax2.set_title('MSX band E image footprint')
# 	plt.show()

# Align("temp_nii_8300_pouss_snrsig2_seuil_sec_test95_avec_seuil_plus_que_0point35_incertitude_moins_de_1000.fits",
#       "maps/reproject/global_widening.fits",
#       "maps/reproject/allooo.fits")

# ----------------------------------------------

# while True:
#     var = input("yo")
#     if var == "1":
#         break

# a = np.array([[1,2], [3,4]])
# print(a **2)

# array = np.array([
#     [1,2,3,4,5,6,3,5],
#     [1,2,3,4,5,6,3,4],
#     [5,2,7,3,6,6,3,2],
#     [4,4,3,3,4,4,4,6],
#     [6,1,6,4,4,3,4,4]
# ])
# i, j = np.indices(array.shape)
# new_array = array[((i-2)**2 + (j-4)**2) <= 4]
                                    
# print(new_array)
# print(new_array.shape)

# ------------------------------------------------------------------

# a = [1,2,3]
# b = [4,5,6]
# print(list(zip((a,b))))

# -----------------------------------------------------------------------------

# global_FWHM = [26.493511278299923, 1.7613582640346546]
# instrumental_function = [19.043028592247367, 0.6296641923412385]
# temperature_map = [5.2978097899979195, 1.610059406400397]
# turbulence_map = [17.64093005817031, 3.8084764530987583]

# # print(np.sqrt(global_FWHM[0]**2-instrumental_function[0]**2-temperature_map[0]**2))
# print((global_FWHM[1]/global_FWHM[0] * 2 * global_FWHM[0]**2 + 
#       instrumental_function[1]/instrumental_function[0] * 2 * instrumental_function[0]**2 + 
#       temperature_map[1]/temperature_map[0] * 2 * temperature_map[0]**2) / turbulence_map[0]**2 * 0.5 *turbulence_map[0])

# ----------------------------------------------------------------------------------------------------------------------------------

# original = Map(fits.open("gaussian_fitting/maps/computed_data/smoothed_instr_f.fits")[0])
# new = Map(fits.open("gaussian_fitting/test_maps/")[0])

# print(original == new)

# file = open("output.txt", "r")
# number = file.readlines()[-1]
# print(number)
# file = open("output.txt", "w")
# file.write(str(number + 1))
# file.close()
# print(file.read())
# file.write(number + 1)
# file.close()

# print(np.random.rand()*100)

# ----------------------------------------------------------------------------------------

# a = fits.open("gaussian_fitting/maps/computed_data/turbulence.fits")[0]
# b = fits.open("gaussian_fitting/maps/computed_data/turbulence_unc.fits")[0]
# b_table = fits.TableHDU(b.data)

# hdu_list = fits.HDUList([
#     a,
#     fits.ImageHDU(b.data, b.header)
# ])

# hdu_list.writeto("gaussian_fitting/test_maps/list_test.fits", overwrite=True)

# file = fits.open("gaussian_fitting/test_maps/list_test.fits")
# print(repr(file[1].header), "\n")
# print(repr(file[0].header))
# plt.imshow(file[1].data)
# plt.show()


# assert 1 == 2, "you're dumb"

# a = 2
# a /= 2 * 4
# print(a)

# import sys
# sys.path.append(".")

# from gaussian_fitting.cube_spectrum import Spectrum
# from gaussian_fitting.fits_analyzer import Map_u


# import scipy
# angstroms_center = 6583.41              # Emission wavelength of NII 
# m = 14.0067 * scipy.constants.u         # Nitrogen mass
# c = scipy.constants.c                   # Light speed
# k = scipy.constants.k                   # Boltzmann constant
# angstroms_FWHM = 2 * np.sqrt(2 * np.log(2)) * angstroms_center * np.sqrt(2000 * k / (c**2 * m))
# speed_FWHM = c * angstroms_FWHM / angstroms_center / 1000
# print(speed_FWHM)

# channels_FWHM = 0.3917525773*10
# spectral_length = 8.60626405229
# wavelength_channel_1 = 6579.48886797
# angstroms_FWHM = channels_FWHM * spectral_length / 48
# angstroms_center = 42.5 * spectral_length / 48 + wavelength_channel_1
# speed_FWHM = scipy.constants.c * angstroms_FWHM / angstroms_center / 1000
# print(speed_FWHM)

# print(0.14376 / 6583.41 * scipy.constants.c)

# a = {"lol": 2}
# a = {"Ploc": 3}
# print(a)

# ----------------------------------------------------------------------------------------------------------------------

# def calc_temp(halpha_width_kms, nii_width_kms):
#     return 4.73*10**4 * ((halpha_width_kms * 1000 * 6562.78 / (scipy.constants.c * (2*np.sqrt(2*np.log(2)))))**2 - (nii_width_kms * 1000 * 6583.41 / (scipy.constants.c * (2*np.sqrt(2*np.log(2)))))**2)

# print(calc_temp(48, 32))


# print(list(zip(range(int(7)), (["OH1", "OH2", "OH3", "OH4", "NII", "Ha"]))))

# print(list((y, [1,2,3,4], "NII") for y in range(10)))

# v = [np.array([1,2])]
# print(np.squeeze(np.array(v)))


# FWHM = 3.1
# angstroms_center = 6717     # Emission wavelength of the element
# m = 32.065 * scipy.constants.u       # Mass of the element
# c = scipy.constants.c                                     # Light speed
# k = scipy.constants.k                                     # Boltzmann constant
# angstroms_FWHM = FWHM * 1000 / c * angstroms_center
# temperature = (angstroms_FWHM * c / angstroms_center)**2 * m / (8 * np.log(2) * k)
# print(temperature)

# lol = [
#     "bateau 0",
#     "bateau 1",
#     "bateau 2",
#     "bateau 3",
#     "bateau 4",
# ]

# for i, j in enumerate(lol):
#     print(i,j)

# for i, j in zip(range(len(lol)), lol):
#     print(i,j)


# v = [5,3]
# j,c = tuple(v)
# print(j)

# a = np.array((1,2,3))
# b = np.tile(a, 2*2).reshape(2,2,3)
# b = np.resize(a, (2,6))
# print(b)
# print(b.shape)

# import time
# lol=0
# start = time.time()
# for i in range(6734025):
#     lol += 1
#     print(".", end="", flush=True)
# stop = time.time()
# print(lol, stop-start)

# a = {1:3}
# a.update()
# print(a)

# a = [[1,2,4],[2,3,5]]
# b = [[3,2],[2,3],[8,9]]
# print(np.ascontiguousarray([a,b]))

# print(np.array_split([1,2,3,4,5,6], 4))

# for key, value in np.array([[1,2],[2,3],[3,4]]):
#     print(key, ":", value)

# string = ".........................................................................................................................................................................................................................................................................................................................X"
# print(len(string))

# a = np.array((5))
# b = np.array((6))
# c = np.array((7,8))
# print(np.append(a,b))
# print(np.append(a,c))

# a = np.array([
#     [1,np.NAN],
#     [3,np.NAN]
# ])
# print(a)
# print(a[0,0][~np.isnan(a[0,0])])
# print(a[np.isnan(a)])


# a = 5.000
# b = 4.494
# print(a%1 == 0)
# print(b%1)

# a = [5,-8,4,-5,8,-4]
# print(np.nanmean(np.sqrt(a)))

# a = np.arange(0, 1, 0.2)
# print(a[(np.abs(a-0.15)).argmin()])
# print(np.append(np.array([]), a))

# print(np.arange(0.1,5.1,0.1))

# s = slice(1,3)
# b = [1,2,3,4,5,6]
# print(b[s])

# print(np.round(np.arange(0.1,1.6,0.1), 1))
# print(np.arange(0.1,1.6,0.1), 1)


# print(np.concatenate(np.array((None)), np.linspace(1,167,167)))

# f = np.array([1,2,3,4])
# print(f[np.array([1,0,2])])

# a = np.array([1,2,3])
# b = np.array([0.1,0.2,0.3])

# print(np.stack((a,b)).transpose())

# a = np.array([1,2])
# print(np.vstack((np.array([[3,4],[5,6]]), a)))

# c = np.array([[1,2],[3,4]])
# print(c)
# print(np.sum(c, axis=1))


# array1 = np.array([1, 2, 3, 4, 5, 6])
# array2 = np.array([[3, 4, 5, 6, 7, 8], [5, 6, 7, 8, 9, 0]])

# Use np.block to concatenate the arrays
# result = np.block([[array1], [array2]])
# print(result)
# print(array2.transpose())

# print(np.array([[1,2,3],[4,5,6]]).shape)


# a = np.array([[1,2],[3,4],[5,6]])
# b = np.array([[7,8],[9,0]])

# print(np.vstack((a,b)))

# def test(f, g):
#     print(f, g)

# test(None)

# boo = False
# a = test("a", "b" if boo else None)

# list1 = [1, 2, 3, 4, 5, 6]
# list2 = ['A', 'B', 'C']

# result = [(list1[i], list2[i // (len(list1) // len(list2))]) for i in range(len(list1))]

# print(result)


# a, b = [2,3]
# print(b,a)

# a = [i for i in range(10)]
# f = (1,4)
# print(a[slice(*f)])


# b = np.array([[1,2,3],[4,5,6]])
# print(b[1,1:3])

# from ..fits_analyzer import Data_cube


# print("\033[1;32mfjdkaj;dsfl")


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Generate a sample 3D data cube
# data_cube = np.random.rand(10, 10, 10)

# # Set up the figure and axis
# fig, ax = plt.subplots()
# image = ax.imshow(data_cube[:, :, 0], cmap='viridis')

# # Define the update function for the animation
# def update(frame):
#     image.set_array(data_cube[:, :, frame])
#     return [image]

# # Create the animation
# num_frames = data_cube.shape[2]
# ani = FuncAnimation(fig, update, frames=num_frames, interval=200, blit=True)

# # Show the animation
# plt.show()


# import numpy as np
# from uncertainties import ufloat

# # Your data and uncertainty arrays
# data_array = np.array([1.0, 2.0, 3.0])
# uncertainty_array = np.array([0.1, 0.2, 0.3])

# # Define a function to convert data and uncertainty to ufloat
# def create_ufloat(value, uncertainty):
#     return ufloat(value, uncertainty)

# # Vectorize the function
# vectorized_ufloat = np.vectorize(create_ufloat)

# # Apply the vectorized function to create ufloat_array
# ufloat_array = vectorized_ufloat(data_array, uncertainty_array)

# print(ufloat_array)


# dataset_1 = np.array((
#     uncertainties.ufloat(1,0.1),
#     uncertainties.ufloat(2,0.2),
#     uncertainties.ufloat(3,0.3),
#     uncertainties.ufloat(4,0.4),
#     uncertainties.ufloat(5,0.5),
#     uncertainties.ufloat(6,0.6),
#     uncertainties.ufloat(7,0.7),
#     uncertainties.ufloat(8,0.8),
#     uncertainties.ufloat(9,0.9),
# ))

# print(np.mean(dataset_1))
# # This gives 5.00+/-0.19

# # But when I calculate it this way
# dataset_2 = np.array((
#     [1,0.1],
#     [2,0.2],
#     [3,0.3],
#     [4,0.4],
#     [5,0.5],
#     [6,0.6],
#     [7,0.7],
#     [8,0.8],
#     [9,0.9]
# ))
# print(np.mean(dataset_2[:,0]), np.mean(dataset_2[:,1]))
# # This gives 5.0+/-0.5

# print(np.std(np.array((
#     0.1,
#     0.2,
#     0.3,
#     0.4,
#     0.5,
#     0.6,
#     0.7,
#     0.8,
#     0.9,
# ))))

# import numpy as np
# import uncertainties
# from uncertainties import unumpy

# # Your first dataset
# dataset_1 = np.array([
#     uncertainties.ufloat(1, 0.1),
#     uncertainties.ufloat(2, 0.2),
#     uncertainties.ufloat(3, 0.3),
#     uncertainties.ufloat(4, 0.4),
#     uncertainties.ufloat(5, 0.5),
#     uncertainties.ufloat(6, 0.6),
#     uncertainties.ufloat(7, 0.7),
#     uncertainties.ufloat(8, 0.8),
#     uncertainties.ufloat(9, 0.9),
# ])

# # Extract nominal values and uncertainties
# nominal_values = unumpy.nominal_values(dataset_1)
# std_dev_values = unumpy.std_devs(dataset_1)

# # Calculate mean and its uncertainty
# mean_value = np.mean(nominal_values)
# uncertainty_value = np.sqrt(np.sum(std_dev_values**2) / len(dataset_1))

# print("Mean: {:.2f}+/-{:.2f}".format(mean_value, uncertainty_value))

import os

def upfold(path, degree):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path.split("/")[:-degree]
    return "/".join(dir_path)



from gaussian_fitting.fits_analyzer import Data_cube

