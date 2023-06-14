from astropy.io import fits
from astropy.wcs import WCS

from fits_analyzer import Data_cube, Map

from reproject import reproject_interp

import matplotlib.pyplot as plt
import numpy as np

# nuit_3 = fits.open("lambda_3.fits")[0].data
# nuit_4 = fits.open("lambda_4.fits")[0].data
# header = fits.open("lambda_3.fits")[0].header

# nuit_34 = np.flip(np.sum((nuit_3, nuit_4), axis=0), axis=(1,2))
# plt.imshow(nuit_34[15,:,:])
# plt.show()
# # fits.writeto("night_34.fits", nuit_34, header, overwrite=True)

# dc = Data_cube()


"""
file_night_34 = fits.open("night_34.fits")

header_1 = (file_night_34[0].header).copy()
header_1["CRPIX1"] = 589
header_1["CRPIX2"] = 477
header_1["CRVAL1"] = (36.7706 + 13 * 60 + 23 * 3600)/(24 * 3600) * 360
header_1["CRVAL2"] = 61 + (30 * 60 + 39.141)/3600
header_1["CDELT1"] = -0.0005168263088 / 2.1458
header_1["CDELT2"] = 0.0002395454546 / 1.0115

header_2 = (file_night_34[0].header).copy()
header_2["CRPIX1"] = 642
header_2["CRPIX2"] = 442
header_2["CRVAL1"] = (30.2434 + 13 * 60 + 23 * 3600)/(24 * 3600) * 360
header_2["CRVAL2"] = 61 + (30 * 60 + 10.199)/3600
header_2["CDELT1"] = -0.0005168263088 / 2.1956
header_2["CDELT2"] = 0.0002395454546 / 0.968

header_3 = (file_night_34[0].header).copy()
header_3["CRPIX1"] = 674
header_3["CRPIX2"] = 393
header_3["CRVAL1"] = (26.2704 + 13 * 60 + 23 * 3600)/(24 * 3600) * 360
header_3["CRVAL2"] = 61 + (29 * 60 + 28.387)/3600
header_3["CDELT1"] = -0.0005168263088 / 2.14
header_3["CDELT2"] = 0.0002395454546 / 0.942

# dc.save_as_fits_file("night_34_3a.fits", file_night_34[0].data, header_3)
"""

"""
file = fits.open("maps/data/fwhm_NII.fits")[0].data
file_u = fits.open("maps/data/fwhm_NII_unc.fits")[0].data
header = fits.open("night_34_tt_e.fits")[0].header

dc.save_as_fits_file("maps/data/fwhm_NII_wcs.fits",
                               file, 
                               dc.bin_header(header, 2))
dc.save_as_fits_file("maps/data/fwhm_NII_unc_wcs.fits",
                               file_u,
                               dc.bin_header(header, 2))
"""

region_1_header = fits.open("night_34_1a.fits")[0].header
region_2_header = fits.open("night_34_2a.fits")[0].header
region_3_header = fits.open("night_34_3a.fits")[0].header
global_header = fits.open("night_34_tt_e.fits")[0].header

fwhm_map = fits.open("maps/data/corrected_fwhm.fits")[0]
fwhm_map_unc = fits.open("maps/data/corrected_fwhm_unc.fits")[0]

region_widening_1 = fits.open("maps/reproject/region_1_widening.fits")[0]
region_widening_2 = fits.open("maps/reproject/region_2_widening.fits")[0]
region_widening_3 = fits.open("maps/reproject/region_3_widening.fits")[0]

cor = fits.open("maps/data/corrected_fwhm.fits")[0]
raw = fits.open("maps/data/fwhm_NII.fits")[0]
cal = Map(fits.open("maps/data/smoothed_instr_f.fits")[0]).bin_map(2)

# test_map = fits.open("test_3a.fits")[0]

# test_3b = Data_cube(fits.open("night_34_3a.fits")[0]).bin_cube(2)

print(np.sum(raw.data - (cor.data + cal.data)))


# print(repr(region_3_header))
# print(repr(test_map.header))

# print(repr(region_1_header))
# wcs = WCS(global_header)
# wcs.sip = None
# wcs = wcs.dropaxis(2)

# print(fwhm_map)

# dc.save_as_fits_file("maps/reproject/global_widening11.fits",
#                                fwhm_map.data, 
#                                dc.bin_header(wcs.to_header(relax=True), 2))

# dc.save_as_fits_file("maps/reproject/global_widening_unc11.fits",
#                                fwhm_map_unc.data, 
#                                dc.bin_header(wcs.to_header(relax=True), 2))

# file_d = fits.open("maps/reproject/global_widening11.fits")[0].data

# m = Map(file_d)
# m.plot_map(m.data, False, (0,40))

# fwhm_region_1 = fits.open("maps/reproject/region_1_widening.fits")[0]
# global_region = fits.open("maps/reproject/global_widening.fits")[0]
# temp_map = fits.open("temp_nii_8300_pouss_snrsig2_seuil_sec_test95_avec_seuil_plus_que_0point35_incertitude_moins_de_1000.fits")[0]

# ax1 = plt.subplot(1,3,1, projection=WCS(fwhm_region_1.header))
# ax1.imshow(fwhm_region_1.data, vmin=0, vmax=40)
# ax2 = plt.subplot(1,3,2)
# ax2.imshow(reproject.reproject_interp(fwhm_region_1, temp_map.header)[0], vmin=0, vmax=40)
# ax3 = plt.subplot(1,3,3)
# ax3.imshow(temp_map.data, vmin=2000, vmax=10000)
# plt.show()

# reprojected_global_map = reproject_interp(global_region, temp_map.header)[0]
# ax1 = plt.subplot(1,3,1, projection=WCS(temp_map.header))
# ax1.imshow(reprojected_global_map, vmin=0, vmax=40, origin="lower")
# ax2 = plt.subplot(1,3,2, projection=WCS(temp_map.header))
# ax2.imshow(global_region.data, vmin=0, vmax=40, origin="lower")
# ax3 = plt.subplot(1,3,3, projection=WCS(temp_map.header))
# ax3.imshow(reproject_interp(global_region, temp_map.header)[1])
# plt.show()

# reprojected_global_map = reproject_interp(temp_map, global_region.header)[0]
# ax1 = plt.subplot(1,1,1, projection=WCS(temp_map.header))
# ax1.imshow(reprojected_global_map, vmin=0, vmax=40, origin="lower")
# plt.show()
