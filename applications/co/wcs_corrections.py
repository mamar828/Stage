import numpy as np
from astropy.io import fits

from src.headers.header import Header
from src.hdu.cubes.cube import Cube
from src.coordinates.ds9_coords import DS9Coords
from src.coordinates.equatorial_coords import RA, DEC


# N2 = Cube.load("data/Loop4_co/N2/Loop4N2_Conv_Med_FinalJS.fits")

# N2.header["CRPIX1"] = 16
# N2.header["CRPIX2"] = 8
# N2.header["CRVAL1"] = RA.from_sexagesimal("8:27:20").degrees
# N2.header["CRVAL2"] = DEC.from_sexagesimal("60:06:00").degrees
# N2.save("data/Loop4_co/N2/Loop4N2_Conv_Med_FinalJS_wcs.fits")


N4 = Cube.load("data/Loop4_co/N4/Loop4N4_Conv_Med_FinalJS.fits")

N4.header["CRPIX1"] = 10
N4.header["CRPIX2"] = 33
N4.header["CRVAL1"] = RA.from_sexagesimal("8:08:24").degrees
N4.header["CRVAL2"] = DEC.from_sexagesimal("61:20:45").degrees
N4.save("data/Loop4_co/N4/Loop4N4_Conv_Med_FinalJS_wcs.fits")
