from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes


from data_cube import Data_cube


class HI_cube(Data_cube):
    """
    Encapsulate the methods useful for shear analysis in HI data.
    """

    def plot_cube(self, z_coordinate: int, color_scale: tuple=(0,25), scatters: list[np.ndarray]=None):
        """
        Plot a Data_cube by taking a slice at a certain z coordinate.

        Arguments
        ---------
        z_coordinate: int. Specifies the z coordinate at which the Data_cube is sliced to be plotted.
        color_scale: tuple, default=(0,25). First element specifies the lower color_scale limit and the second element
        sets the upper limit.
        scatters: list of np.ndarray, optional. Gives the scatters to plot on top of the Data_cube slice.
        """
        fig = plt.figure()
        # The axes are set to have celestial coordinates
        ax = plt.axes()
        # ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=WCS(self.header)[z_coordinate,:,:])
        fig.add_axes(ax)
        # The turbulence map is plotted along with the colorbar, inverting the axes so y=0 is at the bottom
        plt.colorbar(ax.imshow(self.data[z_coordinate,:,:], origin="lower", vmin=color_scale[0], vmax=color_scale[1]))
        # Add the scatters
        if scatters is not None:
            # Plot if a single scatter is given and the element is not a list
            if isinstance(scatters, np.ndarray):
                plt.scatter(*scatters, s=1, c="r", marker=",")
            else:
                for scatter in scatters:
                    plt.scatter(*scatter, s=1, c="r", marker=",")
        plt.show()



class HI_slice:
    """
    Encapsulate the methods useful for analyzing a HI_cube slice.
    """

    def __init__(self, HI_cube: HI_cube, z_coordinate: int, y_limits: list=[130, 350]):
        """
        Initialize a HI_slice object.

        Arguments
        ---------
        HI_cube: HI_cube. Data cube from which the slice is created.
        z_coordinate: int. Specifies the z coordinate at which the Data_cube is sliced.
        y_limits: list. Gives the pixels between which the search will be made. Pixels lower than the lower limit or
        higher than the higher limit will be considered as noise.
        """
        self.data = HI_cube.data[z_coordinate,:,:]
        self.info = HI_cube.info
        self.header = HI_cube.header
        self.y_limits = y_limits

    def plot(self, color_scale: tuple=(0,25), scatters: list[np.ndarray]=None):
        """
        Plot a HI_slice pbject.

        Arguments
        ---------
        color_scale: tuple, default=(0,25). First element specifies the lower color_scale limit and the second element
        sets the upper limit.
        scatters: list of np.ndarray, optional. Gives the scatters to plot on top of the Data_cube slice.
        """
        fig = plt.figure()
        # The axes are set to have celestial coordinates
        ax = plt.axes()
        # ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=WCS(self.header)[z_coordinate,:,:])
        fig.add_axes(ax)
        # The turbulence map is plotted along with the colorbar, inverting the axes so y=0 is at the bottom
        plt.colorbar(ax.imshow(self.data, origin="lower", vmin=color_scale[0], vmax=color_scale[1]))
        # Add the scatters
        if scatters is not None:
            # Plot if a single scatter is given and the element is not a list
            if isinstance(scatters, np.ndarray):
                plt.scatter(*scatters, s=1, c="r", marker=",")
            else:
                for scatter in scatters:
                    plt.scatter(*scatter, s=1, c="r", marker=",")
        plt.show()

    def get_horizontal_maximums(self) -> np.ndarray:
        """
        Get the maximum of each horizontal line.

        Returns
        -------
        np.ndarray: two dimensional array of the x_coordinate of the maximum of each line and value of that maximum, in
        the order of increasing y.
        """
        return np.stack((np.argmax(self.data, axis=1), np.max(self.data, axis=1)), axis=1)
    
    def get_shear_points(self, tolerance: float=0.35) -> list:
        """
        Detect shear by analysis of signal drops.

        Arguments
        ---------
        tolerance: float. Controls the sensitivity of signal drop detection. The value given must be between 0 and 1
        and corresponds to the percentage of the average peak value along each horizontal line (determined with the
        get_horizontal_maximums() method) that will be considered a signal drop. E.g. for a value of 0.5, if a pixel
        along the central line has a value beneath 0.5 times the average peak value, it will be flagged as a potential
        signal drop.

        Returns
        -------
        
        """
        maxs = self.get_horizontal_maximums()
        x_center = scipy.stats.mode(maxs[slice(*self.y_limits),0])[0][0]
        # pixel_offset=3
        # y_shear = np.where(np.abs(maxs[:,0] - x_center) >= pixel_offset)[0]

        average_max = np.mean(maxs[slice(*self.y_limits),1])
        y_shear = np.where(self.data[:,int(x_center)] <= average_max * tolerance)[0]
        print(y_shear[(self.y_limits[0] <= y_shear) & (y_shear <= self.y_limits[1])])
        return y_shear[(self.y_limits[0] <= y_shear) & (y_shear <= self.y_limits[1])]
    
    def get_shear_width(self, ) -> int:
        pass


# n=149
HI = HI_cube(fits.open("HI_regions/LOOP4_bin2.fits")).swap_axes({"x": "v", "y": "l", "z": "b"})
slicy = HI_slice(HI, 149)
slicy.plot(scatters=np.stack((slicy.get_horizontal_maximums()[:,0], np.arange(slicy.data.shape[0])), axis=0))
# maxs = HI.get_horizontal_maximums(n)
# print(np.mean(maxs[150:330]))
# HI.plot_cube(n, scatters=np.stack((maxs, np.arange(HI.data.shape[1])), axis=0))

