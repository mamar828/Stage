import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.cubes.cube import Cube


c = Cube.load("data/loop_co/Loop4N1_FinalJS.fits")[540:790,:,:]


for i in range(310, c.data.shape[1] * c.data.shape[2]):
    s = SpectrumCO(c.data[:,i // c.data.shape[2],i % c.data.shape[2]], c.header)
    if not np.isnan(s.data).all():
        s.fit()
        print(s.fitted_function)
        s.plot_fit(plot_initial_guesses=True, text=f"i={i}", plot_all=True)

# for x in range(c.data.shape[2]):
#     for y in range(c.data.shape[1]):
#         s = SpectrumCO(c.data[:,y,x], c.header)
#         if not np.isnan(s.data).all():
#             s.fit()
#             print(s.fitted_function)
#             s.plot_fit(plot_initial_guesses=True, text=f"(x,y)=({x},{y})", plot_all=True)
        


# Test bin algorithm

