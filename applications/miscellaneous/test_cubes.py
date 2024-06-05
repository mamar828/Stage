import matplotlib.pyplot as plt


from src.hdu.cubes.cube import Cube


fig, ax = plt.subplots(1)
c = Cube.load("data/Loop4_co/p/Loop4p_Conv_Med_FinalJS.fits")

# c = c.invert_axis(2)
# c.save("og.fits")
# a = c.data.plot(fig, ax)
# plt.show()

s = c.swap_axes(0,1).swap_axes(0,2)
s.save("Loop4p_swapped.fits")
