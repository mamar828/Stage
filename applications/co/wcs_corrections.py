import numpy as np
from astropy.io import fits

from src.headers.header import Header
from src.hdu.cubes.cube import Cube
from src.coordinates.ds9_coords import DS9Coords
from src.coordinates.equatorial_coords import RA, DEC


# N2 = Cube.load("data/Loop4_co/N2/Loop4N2_Conv_Med_FinalJS_DEPRECATED.fits")

# N2.header["CRPIX1"] = 16
# N2.header["CRPIX2"] = 8
# N2.header["CRVAL1"] = RA.from_sexagesimal("8:27:20").degrees
# N2.header["CRVAL2"] = DEC.from_sexagesimal("60:06:00").degrees
# N2.save("data/Loop4_co/N2/Loop4N2_Conv_Med_FinalJS_wcs.fits")


# N4 = Cube.load("data/Loop4_co/N4/Loop4N4_Conv_Med_FinalJS_DEPRECATED.fits")
# # A new value for CDELT1 is needed as the scaling is not applied properly
# # The delta between N4W10S2 and N4S2 is used, at DEC=61:17:45
# CDELT1 = (RA.from_sexagesimal("8:06:22").degrees - RA.from_sexagesimal("8:08:24").degrees) / (10*3) \
#        * np.cos(np.radians(DEC.from_sexagesimal("61:17:45").degrees))
# # The delta between N4NW and N4WS9 is used
# CDELT2 = (DEC.from_sexagesimal("61:22:15").degrees - DEC.from_sexagesimal("61:07:15").degrees) / (10*3)
# center_pixel = (10,33)
# N4.header["CDELT1"] = CDELT1
# N4.header["CDELT2"] = CDELT2
# N4.header["CRPIX1"] = 10
# N4.header["CRPIX2"] = 33
# N4.header["CRVAL1"] = RA.from_sexagesimal("8:08:24").degrees
# N4.header["CRVAL2"] = DEC.from_sexagesimal("61:20:45").degrees
# N4.save("data/Loop4_co/N4/Loop4N4_Conv_Med_FinalJS_wcs_2.fits")


# p = Cube.load("data/Loop4_co/p/Loop4p_Conv_Med_FinalJS_DEPRECATED.fits")

# p.header["CRPIX1"] = 36
# p.header["CRPIX2"] = 41
# p.header["CRVAL1"] = RA.from_sexagesimal("8:05:32").degrees
# p.header["CRVAL2"] = DEC.from_sexagesimal("60:32:30").degrees
# p.save("data/Loop4_co/p/Loop4p_Conv_Med_FinalJS_wcs.fits")
