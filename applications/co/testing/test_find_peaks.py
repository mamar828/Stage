import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks

from src.spectrums.spectrum_co import SpectrumCO
from src.hdu.cubes.cube import Cube


# c = Cube.load("data/external/loop_co/Loop4N1_FinalJS.fits")[500:800,:,:]
c = Cube.load("data/external/loop_co/Loop4N2_Conv_Med_FinalJS.fits")[500:800,:,:]
# c = Cube.load("data/external/loop_co/Loop4N4_Conv_Med_FinalJS.fits")[500:850,:,:] #sigmas_threshold=7, minimum_peak_distance=6
# c = Cube.load("data/external/loop_co/Loop4p_Conv_Med_FinalJS.fits")[550:900,:,:]

# for x in range(c.data.shape[2]):
#     for y in range(c.data.shape[1]):
#         s = SpectrumCO(c.data[:,y,x], c.header)
for i in range(495, c.data.shape[1] * c.data.shape[2]):
    s = SpectrumCO(c.data[:,i // c.data.shape[2],i % c.data.shape[2]], c.header)
    if not np.isnan(s.data).all():
        print(i)
        s.fit()
        # print(s.fitted_function)

        
        peaks, _ = find_peaks(s.data, distance=20)
        peaks2, _ = find_peaks(s.data, prominence=1)      # BEST!
        peaks3, _ = find_peaks(s.data, width=20)
        peaks4, _ = find_peaks(s.data, threshold=0.4)     # Required vertical distance to its direct neighbouring samples, pretty useless
        # plt.subplot(2, 2, 1)
        # plt.plot(peaks, s.data[peaks], "xr"); plt.plot(s.data); plt.legend(['distance'])
        # plt.subplot(2, 2, 2)
        # plt.plot(peaks2, s.data[peaks2], "ob"); plt.plot(s.data); plt.legend(['prominence'])
        # plt.subplot(2, 2, 3)
        # plt.plot(peaks3, s.data[peaks3], "vg"); plt.plot(s.data); plt.legend(['width'])
        # plt.subplot(2, 2, 4)
        # plt.plot(peaks4, s.data[peaks4], "xk"); plt.plot(s.data); plt.legend(['threshold'])
        # plt.show()
        fig, axs = plt.subplots(2)
        p_1 = find_peaks(s.data, prominence=1)[0]
        p_2 = find_peaks(s.data, prominence=0.7, height=float(np.std(s.data[:100]) * 6), distance=10)[0]
        axs[0].plot(p_1, s.data[p_1], "ob")
        axs[0].plot(s.data)
        axs[1].plot(p_2, s.data[p_2], "ob")
        axs[1].plot(s.data)
        fig.suptitle(i)
        plt.show()
