import numpy as np

from src.hdu.cubes.cube import Cube


c = Cube.load("data/Loop4_co/N4/Loop4N4_Conv_Med_FinalJS.fits")

s = c.swap_axes(0,1).swap_axes(0,2)
s.save("data/Loop4_co/N4/Loop4N4_swapped.fits")
s = s.swap_axes(0,2).swap_axes(0,1)
s.data[0,:,:] = np.full_like(s.data[np.newaxis,0,:,:], np.NAN)
s.save("data/Loop4_co/N4/t.fits")
