import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from src.regions.mask import Mask


mask = Mask(image_shape=(1000, 1000))

# circles = [mask.circle((i,i), 50) for i in np.linspace(100, 900, 10)]

# plt.imshow(sum(circles))
# plt.show()


# circles = [mask.circle((250, 500), 300), mask.circle((750, 500), 300)]

# plt.imshow(circles[0] & circles[1])
# plt.show()
# plt.imshow(circles[0] | circles[1])
# plt.show()
# plt.imshow(circles[0] ^ circles[1])
# plt.show()

# ell = mask.ellipse((400,500), 200, 100, 15)
# plt.imshow(ell)
# plt.show()

# rect = mask.rectangle((500,500), 200, 100, 20)
# plt.imshow(rect)
# plt.show()

# pol = mask.polygon([(50,500), (950,500), (500,250)])
# plt.imshow(pol)
# plt.show()

# inner_cir = mask.circle((350,350), 100)
# outer_cir = mask.circle((350,350), 150)

# plt.imshow(inner_cir ^ outer_cir)
# plt.show()


# m = Mask((512,512))
# test_map = fits.open("summer_2023/gaussian_fitting/maps/computed_data/NII_mean.fits")[0]
# mask = m.open_as_image_coord("summer_2023/gaussian_fitting/regions/region_1.reg", test_map.header)
# plt.colorbar(plt.imshow(test_map.data))
# plt.clim(13,17)
# plt.show()
# plt.colorbar(plt.imshow(mask))
# plt.show()
# masked_map = mask * test_map.data
# plt.colorbar(plt.imshow(masked_map))
# plt.clim(13,17)
# plt.show()
# plt.colorbar(plt.imshow(mask ^ pol))
# plt.clim(13,17)
# plt.show()


# WCS regions

SHAPE_X, SHAPE_Y = 512, 512
FITS_FILENAME = "summer_2023/gaussian_fitting/maps/computed_data/NII_mean.fits"
REG_FILENAME = "summer_2023/gaussian_fitting/regions/region_1.reg"

m = Mask((SHAPE_Y,SHAPE_X))

test_fits = fits.open(FITS_FILENAME)[0]
mask = m.open_as_image_coord(REG_FILENAME, test_fits.header)
plt.imshow(mask)
plt.title("mask")
plt.show()
masked_fits = mask * test_fits.data
plt.colorbar(plt.imshow(masked_fits))
plt.title("masked_fits")
# plt.clim()            # adjust as needed
plt.show()
