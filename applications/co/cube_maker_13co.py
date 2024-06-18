import numpy as np
from astropy.io import fits

from src.headers.header import Header
from src.hdu.cubes.cube import Cube
from src.coordinates.ds9_coords import DS9Coords


N1_12CO = Cube.load("data/Loop4_co/N1/Loop4N1_FinalJS.fits")
N1_13CO = Cube.load("data/Loop4_co/13co/felix/cubeN113_zero.fits")
N1_13CO.data[N1_13CO.data == 0] = np.NAN
N1_13CO.save("data/Loop4_co/N1/Loop4N1_13co.fits")

N2_12CO = Cube.load("data/Loop4_co/N2/Loop4N2_Conv_Med_FinalJS.fits")
N2_13CO = Cube.load("data/Loop4_co/13co/felix/cubeN113_zero.fits")
N2_13CO.data[N2_13CO.data == 0] = np.NAN
N2_13CO.save("data/Loop4_co/N2/Loop4N2_13co.fits")

N4_12CO = Cube.load("data/Loop4_co/N4/Loop4N4_Conv_Med_FinalJS.fits")
N4_13CO = Cube.load("data/Loop4_co/13co/felix/cubeN413_zero.fits")
N4_13CO.data[N4_13CO.data == 0] = np.NAN
N4_13CO.save("data/Loop4_co/N4/Loop4N4_13co.fits")

p_12CO = Cube.load("data/Loop4_co/p/Loop4p_Conv_Med_FinalJS.fits")
p_13CO = Cube.load("data/Loop4_co/3_co/felix/cubep13_zero.fits")
p_13CO.data[p_13CO.data == 0] = np.NAN
p_13CO.save("data/Loop4_co/p/Loop4p_13co.fits")
