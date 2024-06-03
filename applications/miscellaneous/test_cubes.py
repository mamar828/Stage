import matplotlib.pyplot as plt
from src.hdu.cubes.cube import Cube


fig, ax = plt.subplots(1)
c = Cube.load("data/Loop4_co/p/Loop4p_Conv_Med_FinalJS.fits").invert_axis(2)
c.save("nog.fits")
raise
a = c.data.plot(fig, ax)
plt.show()
