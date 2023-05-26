import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator



def plot_spectrum():
    x, y = 181, 133

    data = fits.open("cube_NII_Sh158_with_header.fits")[0].data
    y_values = data[:,y-1,x-1]
    x_values = np.arange(1,49)

    plt.plot(x_values, y_values)
    plt.show()


def plot_map(values=None):
    if values.shape == None:
        data = fits.open("cube_NII_Sh158_with_header.fits")[0].data
        values = data[41,:,:]
    plt.imshow(values, cmap="gist_earth", origin="lower")
    plt.show()


def bin_map(data, nb_pix_bin):
    try:
        bin_array = data.reshape(int(data.shape[0]/nb_pix_bin), nb_pix_bin, int(data.shape[1]/nb_pix_bin), nb_pix_bin)
    except ValueError:
        raise IndexError("the array provided can not be divided by the nb_pix_bin specified")
    new_values = bin_array.sum(axis=(1,3))
    plot_map(new_values)


ds9_data = fits.open("cube_NII_Sh158_with_header.fits")[0].data

bin_map(ds9_data[13,:300,:300], 1)

