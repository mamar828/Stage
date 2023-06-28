import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
import os
from astropy.io import fits
import scipy


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

def calc_temp(halpha_width_kms, nii_width_kms):
    return 4.73*10**4 * ((halpha_width_kms * 1000 * 6562.78 / scipy.constants.c)**2 - (nii_width_kms * 1000 * 6583.41 / scipy.constants.c)**2)

print(calc_temp(33.927336320324216, 30.524020241208408))
