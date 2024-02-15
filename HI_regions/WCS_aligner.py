import numpy as np

from astropy.io import fits
from data_cube import Data_cube
from coords import l, b


Spider = Data_cube(fits.open("HI_regions/data_cubes/spider/Spider_bin4.fits"))

# Rename types
Spider.header["CTYPE1"] = ("GLON-SFL", "Conversion was manually made from RA---NCP")
Spider.header["CTYPE2"] = ("GLAT-SFL", "Conversion was manually made from DEC--NCP")

# First corrections
Spider.header["CRVAL1"] = str(l("134:59:54.273"))
Spider.header["CRVAL2"] = str(b("40:00:02.417"))
Spider.header["CDELT1"] = l("135:00:58.632").sexagesimal - l("134:59:50.252").sexagesimal
Spider.header["CDELT2"] = b("39:49:41.718").sexagesimal - b("39:50:35.149").sexagesimal

# Rotate cube
Spider.header["CROTA1"] = -41.6085*0.99739

# Fine-tuning
Spider.header["CDELT1"] *= 1.052
Spider.header["CDELT2"] *= 1.347

Spider.save_as_fits_file("HI_regions/data_cubes/spider/Spider_bin4_galactic.fits", overwrite=True)
