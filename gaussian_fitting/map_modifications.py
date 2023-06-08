from astropy.io import fits

from cube_analyzer import Data_cube_analyzer


# nuit_3 = fits.open("lambda_3.fits")[0].data
# nuit_4 = fits.open("lambda_4.fits")[0].data
# header = fits.open("lambda_3.fits")[0].header

# nuit_34 = np.flip(np.sum((nuit_3, nuit_4), axis=0), axis=(1,2))
# plt.imshow(nuit_34[15,:,:])
# plt.show()
# # fits.writeto("night_34.fits", nuit_34, header, overwrite=True)



file = fits.open("night_34.fits")
a = Data_cube_analyzer()

header_1 = (file[0].header).copy()
header_1["CRPIX1"] = 589
header_1["CRPIX2"] = 477
header_1["CRVAL1"] = (36.7706 + 13 * 60 + 23 * 3600)/(24 * 3600) * 360
header_1["CRVAL2"] = 61 + (30 * 60 + 39.141)/3600
header_1["CDELT1"] = -0.0005168263088 / 2.1458
header_1["CDELT2"] = 0.0002395454546 / 1.0115

header_2 = (file[0].header).copy()
header_2["CRPIX1"] = 642
header_2["CRPIX2"] = 442
header_2["CRVAL1"] = (30.2434 + 13 * 60 + 23 * 3600)/(24 * 3600) * 360
header_2["CRVAL2"] = 61 + (30 * 60 + 10.199)/3600
header_2["CDELT1"] = -0.0005168263088 / 2.1956
header_2["CDELT2"] = 0.0002395454546 / 0.968

header_3 = (file[0].header).copy()
header_3["CRPIX1"] = 674
header_3["CRPIX2"] = 393
header_3["CRVAL1"] = (26.2704 + 13 * 60 + 23 * 3600)/(24 * 3600) * 360
header_3["CRVAL2"] = 61 + (29 * 60 + 28.387)/3600
header_3["CDELT1"] = -0.0005168263088 / 2.14
header_3["CDELT2"] = 0.0002395454546 / 0.942

# a.save_as_fits_file("night_34_3a.fits", file[0].data, header_3)
