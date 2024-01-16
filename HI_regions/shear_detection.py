from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes


from data_cube import Data_cube
from galactic_coords import b


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
    
    def extract_shear(
            self,
            l_bounds: list,
            b_bounds: list[b],
            tolerance: float,
            max_regroup_separation: int,
            pixel_width: int=1,
            max_accepted_shear: int=None,
        ) -> dict:
        """
        Extract the shear's data between the given bounds using the provided parameters.

        Arguments
        ---------
        l_bounds: list of pixel numbers. Specifies the bounds between which the search will be made. Both values are
            included.
        b_bounds: list of b objects. Specifies the galactic latitude bounds between which the search will be made. Both
            values are included.
        tolerance: float. Controls the sensitivity of signal drop detection. The value given corresponds to the
            percentage of the average peak value along each horizontal line (determined with the get_horizontal_maximums()
            method) that will be considered a signal drop. E.g. for a value of 0.5, if a pixel along the central line has a
            value beneath 0.5 times the average peak value, it will be flagged as a potential signal drop.
        max_regroup_separation: int. Maximum separation of two consecutive points that will be considered to belong to
            the same signal drop section. This controls how many different regions will be outputted and will merge those
            that are close.
        pixel_width: int, default=1. This parameter must be greater than or equal to 1 and specifies the width of the
            search along every longitude. For example, pixel_width=3 will search along the pixel at v=0 and the two pixels
            to its right. The default value will only search along v=0.
        max_accepted_shear: int, optional. Maximum number of pixels to the left of v=0 that are analyzed to look for a
            maximum.

        Returns
        -------
        dict: each key is the pixel at which the shear was detected and the corresponding value is the data of the
        detected shear outputted by the HI_slice.check_shear() method. 0: bounds of the detected shear in pixels,
        1: shear width in km/s, 2: coordinates of the max shear point.
        """
        # Verify provided arguments
        assert isinstance(pixel_width, int), "pixel_width provided must be an integer"
        assert pixel_width >= 1, "pixel_width must be greater than or equal to 1."

        collected_info = {}
        if self.info["z"] == "b":
            # Convert bounds to array indices
            for b in range(b_bounds[0].to_pixel(self.header), b_bounds[1].to_pixel(self.header)):
                collected_info[b] = HI_slice(self, b, l_bounds).check_shear(
                                                  (max_accepted_shear, tolerance, max_regroup_separation, pixel_width))
        elif self.info["z"] == "l":
            # Convert bounds to array indices
            for l in range(b_bounds[0].to_pixel(self.header), b_bounds[1].to_pixel(self.header)):
                collected_info[b] = HI_slice(self, l, b_bounds).check_shear(
                                                                      (tolerance, max_regroup_separation, pixel_width))
        else:
            raise TypeError("HI_cube should be a rotated cube with either longitude or latitude as z axis.")

        return collected_info
    
    def watch_shear(self, shear_info: dict, fullscreen: bool=False):
        """
        Watch the shear detected by the HI_cube.extract_shear() method.
        
        Arguments
        ---------
        shear_info: dict. Information that should be displayed,
        fullscreen: bool, default=False. Specify if the plot should be opened in full screen.
        """
        for key, value in shear_info.items():
            for bounds, shear_width, max_coords in value:
                current_slice = HI_slice(self, key)
                current_slice.plot(bounds, shear_width, max_coords, fullscreen)
        print("Shear watching ended successfully.")



class HI_slice:
    """
    Encapsulate the methods useful for analyzing a HI_cube slice.
    """

    def __init__(self, HI_cube: HI_cube, z_coordinate: int, y_limits: list=None):
        """
        Initialize a HI_slice object.

        Arguments
        ---------
        HI_cube: HI_cube. Data cube from which the slice is created.
        z_coordinate: int. Specifies the z coordinate at which the Data_cube is sliced.
        y_limits: list. Gives the pixels between which the search will be made. Pixels lower than the lower limit or
            higher than the higher limit will be considered as noise. Both values are included.
        """
        self.data = HI_cube.data[z_coordinate,:,:]
        self.z_coordinate = z_coordinate
        self.info = HI_cube.info
        self.header = HI_cube.header
        if y_limits:
            # Account for the fact that both values are included
            self.y_limits = y_limits[0], y_limits[1]+1

    def plot(self, bounds: list=None, shear_width: float=None, max_coords: tuple=None, fullscreen: bool=False):
        """
        Plot a HI_slice object.

        Arguments
        ---------
        bounds: list, default=None. y limits of the detected shear.
        shear_width: float, default=None. Width of the shear calculated for this specific region.
        max_coords: tuple, default=None. Coordinates of the maximum used for shear_width calculation.
        fullscreen: bool, default=False. Specify if the plot should be opened in full screen.
        """
        color_scale = 0, 25
        fig = plt.figure()
        # The axes are set to have celestial coordinates
        # ax = plt.axes()
        ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=WCS(self.header)[self.z_coordinate,:,:])
        fig.add_axes(ax)
        # The turbulence map is plotted along with the colorbar, inverting the axes so y=0 is at the bottom
        plt.colorbar(ax.imshow(self.data, origin="lower", vmin=color_scale[0], vmax=color_scale[1]))
        
        # Set parameters
        if bounds and shear_width and max_coords:
            min_xlim = self.data.shape[1]/2 - 10
            max_xlim = self.data.shape[1]/2 + max(10, max_coords[0] - self.data.shape[1]/2 + 10)
            plt.xlim(min_xlim, max_xlim)
            plt.ylim(bounds[0]-40, bounds[1]+40)

            # Add the rectangle around the detected shear and enlarge so that the border is at the outside of the
            # concerned pixels
            region = mpl.patches.Rectangle(
                (self.header["CRPIX1"]-0.5,bounds[0]-0.5), 
                max_coords[0] - self.header["CRPIX1"] + 1, bounds[1] - bounds[0] + 1,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            max_shear = mpl.patches.Rectangle(
                (max_coords[0] - 0.5, max_coords[1] - 0.5), 1, 1, linewidth=1, edgecolor='w', facecolor="none"
            )
            ax.add_patch(region)
            ax.add_patch(max_shear)
            z = self.z_coordinate
            plt.title(
                f"Current z_coordinate: {z} ({b.from_pixel(z, self.header)}), shear_width: {shear_width:.2f} km/s"
            )
        
        if fullscreen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        plt.show()
    
    def check_shear(self, params: tuple) -> list:
        """
        Extract potential shear zones and their maximum shear value in km/s.

        Arguments
        ---------
        params: tuple. max_accepted_shear, tolerance, max_regroup_separation, pixel_width. See HI_cube.extract_shear()
            for more informations.

        Returns
        -------
        list: data of every detected shear. Each element has 0: bounds of the detected shear in pixels, 1: shear width
        in km/s, 2: coordinates of the max shear point.
        """
        max_accepted_shear, tolerance, max_regroup_separation, pixel_width = params

        shear_bounds = self.get_point_groups(self.get_signal_drops(tolerance, pixel_width), max_regroup_separation)

        shear_data = []
        for bounds in shear_bounds:
            shear_data.append((bounds, *self.get_max_shear_width(bounds, max_accepted_shear)))
        return shear_data

    def get_horizontal_maximums(self) -> np.ndarray:
        """
        Get the maximum of each horizontal line.

        Returns
        -------
        np.ndarray: two dimensional array of the x_coordinate of the maximum of each line and value of that maximum, in
            the order of increasing y.
        """
        return np.stack((np.argmax(self.data, axis=1), np.max(self.data, axis=1)), axis=1)
    
    def get_point_groups(self, points: list, max_regroup_separation: int) -> list:
        """
        Give the bounds of every group in a list.

        Arguments
        ---------
        points: list. Data that needs to be grouped.
        max_regroup_separation: int. Maximum separation of two consecutive points that will be considered to belong to
            the same signal drop section. This controls how many different regions will be outputted and will merge those
            that are close.
        
        Returns
        -------
        list: each element is a list containing the bounds of a group.
        """
        groups = [[points[0], points[0]]]
        for point in points[1:]:
            if groups[-1][1] + max_regroup_separation >= point:
                groups[-1][1] = point
            else:
                groups.append([point, point])
        return groups
    
    def get_signal_drops(self, tolerance: float, pixel_width: int=1) -> list:
        """
        Get every signal drop along a line from v=0 to the pixel width.

        Arguments
        ---------
        tolerance: float. Controls the sensitivity of signal drop detection. The value given corresponds to the
            percentage of the average peak value along each horizontal line (determined with the get_horizontal_maximums()
            method) that will be considered a signal drop. E.g. for a value of 0.5, if a pixel along the central line has a
            value beneath 0.5 times the average peak value, it will be flagged as a potential signal drop.
        pixel_width: int, default=1. This parameter must be greater than or equal to 1 and specifies the width of the
            search along every longitude. For example, pixel_width=3 will search along the pixel at v=0 and the two pixels
            to its right. The default value will only search along v=0.

        Returns
        -------
        list: every element is a list that contains the bounds of the drop detected.
        """
        # Extract useful informations
        maxs = self.get_horizontal_maximums()
        x_center = int(self.header["CRPIX1"])
        y_slice = slice(*self.y_limits)
        average_max = np.mean(maxs[y_slice,1])

        # Get the array representing every y value that has a drop
        y_shear = np.where(np.any(self.data[:,x_center:x_center+pixel_width] <= average_max * tolerance, axis=1))[0]
        # Return only the values between the y_limits
        return y_shear[(self.y_limits[0] <= y_shear) & (y_shear <= self.y_limits[1])]

    def get_max_shear_width(self, bounds: list, max_accepted_shear: int) -> tuple:
        """
        Get the maximum shear width of a certain region by computing the maximum distance between v=0 and the intensity
        peaks in a certain region in addition to its coordinates.

        Arguments
        ---------
        bounds: list. Values between which the search for the farthest maximum should be made.
        max_accepted_shear: int. Maximum number of pixels to the left of v=0 that are analyzed to look for a maximum.

        Returns
        -------
        tuple: first element is the width in km/s computed using the header's informations and the second element is a
        tuple of the detected maximum's coordinates.
        """
        maxs = self.get_horizontal_maximums()[slice(bounds[0], bounds[1]+1),0]
        if max_accepted_shear:
            print(self.z_coordinate)
            print(maxs)
            np.place(maxs, maxs > max_accepted_shear + self.header["CRPIX1"], 0)
            print(maxs)

        # Compute the distance using the header and convert m/s to km/s
        max_width = - (np.max(maxs) - self.header["CRPIX1"]) * self.header["CDELT1"] / 1000
        coords = np.max(maxs), np.argmax(maxs) + bounds[0]
        return max_width, coords



# n=149
HI = HI_cube(fits.open("HI_regions/LOOP4_bin2.fits")).swap_axes({"x": "v", "y": "l", "z": "b"})
# slicy = HI_slice(HI, 149)
shear_points = HI.extract_shear([130,350], [b("32:30:29.077"),b("32:31:41.077")], 0.4, 10, 3, 7)
# shear_points = HI.extract_shear([130,350], [b("32:30:29.077"),b("32:31:41.077")], 0.4, 10, 3)
HI.watch_shear(shear_points, fullscreen=True)

# a = b("32:30:29.077")
# pix = a.to_pixel(HI.header)
# print(pix)
# print(b.from_pixel(pix, HI.header))

# print(slicy.extract_shear(0.4, 10, 3))
# print(b("31:01:05.078").to_pixel(HI.header))
# print(slicy.get_signal_drops(0.6,3))
# print(slicy.get_point_groups(slicy.get_signal_drops(0.6,3), 10))
# print(slicy.get_max_shear_width(slicy.get_point_groups(slicy.get_signal_drops(0.6,3), 10)[-1]))
# slicy.plot(scatters=np.stack((slicy.get_horizontal_maximums()[:,0], np.arange(slicy.data.shape[0])), axis=0))
# maxs = HI.get_horizontal_maximums(n)
# print(np.mean(maxs[150:330]))
# HI.plot_cube(n, scatters=np.stack((maxs, np.arange(HI.data.shape[1])), axis=0))




# Add limits in galactic coords
# View method

