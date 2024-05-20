from __future__ import annotations

import numpy as np
import matplotlib as mpl
import scipy
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes


from data_cube import Data_cube
from coords import *

from eztcolors import Colors as C



class HI_cube(Data_cube):
    """
    Encapsulate the methods useful for shear analysis in HI data.
    """
    
    def extract_shear(
            self,
            y_bounds: list,
            z_bounds: list,
            **kwargs
        ) -> dict:
        """
        Extract the shear's data between the given bounds using the provided parameters.

        Arguments
        ---------
        y_bounds: list of pixel numbers. Specifies the bounds between which the search will be made. Both values are
            included.
        z_bounds: list of b objects or pixel numbers. Specifies the z bounds between which the search will be made.
            Both values are included.
        kwargs. See LOOP4_slice.check_shear or Spider_slice.check_shear depending on the HI_cube's slice_type for more
            information.

        Returns
        -------
        dict: each key is the pixel at which the shear was detected and the corresponding value is the data of the
            detected shear outputted by the HI_slice.check_shear() method. 0: bounds of the detected shear in pixels,
            1: shear width in km/s, 2: coordinates of the max shear point.
        """
        collected_info = {}
        if self.info["z"] == "b":
            # Convert bounds to array indices
            if isinstance(z_bounds[0], b):
                z_bounds_array = z_bounds[0].to_pixel(self.header), z_bounds[1].to_pixel(self.header) + 1
            elif isinstance(z_bounds[0], int):
                z_bounds_array = z_bounds[0], z_bounds[1] + 1
            else:
                raise NotImplementedError(C.RED + f"z_bounds type ({type(z_bounds)}) not supported." + C.END)
            for z in range(*z_bounds_array):
                # Filter no detected shear
                current_info = self.slice_type(self, z, y_bounds).check_shear(**kwargs)
                if current_info is not None:
                    collected_info[z] = current_info

        elif self.info["z"] == "l":
            # Convert bounds to array indices
            for l in range(z_bounds[0].to_pixel(self.header), z_bounds[1].to_pixel(self.header)+1):
                # Filter no detected shear
                current_info = self.slice_type(self, l, z_bounds).check_shear(**kwargs)
                if current_info is not None:
                    collected_info[b] = current_info

        else:
            raise TypeError("HI_cube should be a rotated cube with either longitude or latitude as z axis.")

        return collected_info
    
    def watch_shear(self, shear_info: dict, fullscreen: bool=False):
        """
        Watch the shear detected by the HI_cube.extract_shear() method.
        
        Arguments
        ---------
        shear_info: dict. Information that should be displayed, in the format outputted by HI_cube.extract_shear():
            keys are the z_coordinates and values are 0: bounds in pixels of the detected shear, 1: shear width in km/s, 
            2: coordinates of the max shear point.
        fullscreen: bool, default=False. Specify if the plot should be opened in full screen.
        """
        for key, value in shear_info.items():
            try:
                for bounds, shear_width, max_coords in value:
                    current_slice = self.slice_type(self, key)
                    current_slice.plot(bounds, shear_width, max_coords, fullscreen)
            except Exception:
                raise Exception(f"{C.RED}{C.BOLD}Exception occured at z={key}.{C.END}")
        
        print(f"{C.GREEN}{C.BOLD}Shear watching ended successfully.{C.END}")



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
        # Slicing starts at 0 whereas DS9 numbering starts at 1
        self.data = HI_cube.data[z_coordinate-1,:,:]
        self.z_coordinate = z_coordinate
        self.info = HI_cube.info
        self.header = HI_cube.header
        self.x_center = int(np.round(self.header["CRPIX1"] - self.header["CRVAL1"] / self.header["CDELT1"]))
        if y_limits:
            # Account for the fact that both values are included
            self.y_limits = y_limits[0], y_limits[1]+1
        else:
            self.y_limits = 0, self.data.shape[1]

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
            the same signal drop section. This controls how many different regions will be outputted and will merge
            those that are close.
        
        Returns
        -------
        list: each element is a list containing the bounds of a group.
        """
        if list(points) != []:
            groups = [[points[0], points[0]]]
            for point in points[1:]:
                if groups[-1][1] + max_regroup_separation >= point:
                    groups[-1][1] = point
                else:
                    groups.append([point, point])
            return groups


class LOOP4_slice(HI_slice):
    """ 
    Encapsulate the methods specific to the analysis of slices of LOOP4 data cubes.
    """

    def plot(self, bounds: list=None, shear_width: float=None, max_coords: tuple=None, fullscreen: bool=False):
        """
        Plot the object in a two-dimensional plot.

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
        header_copy = self.header.copy()
        header_copy["CDELT1"] /= 1000
        ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=WCS(header_copy)[self.z_coordinate,:,:])
        ax.set_xlabel(r"Speed [$\frac{\rm{km}}{\rm{s}}$]")
        fig.add_axes(ax)
        # The turbulence map is plotted along with the colorbar, inverting the axes so y=0 is at the bottom
        plt.colorbar(ax.imshow(self.data, origin="lower", vmin=color_scale[0], vmax=color_scale[1]))
        
        # Set parameters
        if bounds is not None and shear_width is not None and max_coords is not None:
            min_xlim = self.x_center - 10
            max_xlim = self.x_center + max(10, max_coords[0] - self.x_center + 10)
            plt.xlim(min_xlim, max_xlim)
            plt.ylim(bounds[0]-40, bounds[1]+40)

            # Add the rectangle around the detected shear and enlarge so that the border is at the outside of the
            # concerned pixels
            region = mpl.patches.Rectangle(
                (self.x_center - 0.5,bounds[0] - 0.5), 
                max_coords[0] - self.x_center + 1, bounds[1] - bounds[0] + 1,
                linewidth=1, edgecolor='r', facecolor='none'
            )
            # Add the rectangle around the max shear detected
            max_shear = mpl.patches.Rectangle(
                (max_coords[0] - 0.5, max_coords[1] - 0.5), 1, 1, linewidth=1, edgecolor='w', facecolor="none"
            )
            ax.add_patch(region)
            ax.add_patch(max_shear)
            z = self.z_coordinate

            # Set info parameters
            if "GLAT" in self.header["CTYPE3"]:
                plt.title(
                    f"Current z_coordinate: {z} ({b.from_pixel(z, self.header).to_clock()}), " + 
                    f"shear_width: {shear_width:.2f} km/s"
                )
            else:
                plt.title(
                    f"Current z_coordinate: {z}, shear_width: {shear_width:.2f} km/s"
                )
        
        if fullscreen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        plt.show()

    def check_shear(self, *,
            tolerance: float,
            accepted_width: int=None,
            max_regroup_separation: int=0,
            pixel_width: int=1,
            max_accepted_shear: int=None
        ) -> list:
        """
        Extract potential shear zones and their maximum shear value in km/s.

        Arguments
        ---------
        tolerance: float. Controls the sensitivity of signal drop detection. The value given corresponds to the
            percentage of the average peak value along each horizontal line (determined with the
            get_horizontal_maximums method) that will be considered a signal drop. E.g. for a value of 0.5, if a
            pixel along the central line has a value beneath 0.5 times the average peak value, it will be flagged as a
            potential signal drop.
        accepted_width: int, default=None. Normal width of the HI's central band. Greatest accepted distance, in pixels,
            between the HI's center and the maximum of a horizontal line that will not be flagged as a shear point. For
            example, a value of 0 will remove shears of 0 pixels and a value of 1 will remove shears of 1 pixels. Use
            None to accept all shears.
        max_regroup_separation: int, default=0. Maximum separation of two consecutive points that will be considered to
            belong to the same signal drop section. This controls how many different regions will be outputted and will
            merge those that are close.
        pixel_width: int, default=1. This parameter must be greater than or equal to 1 and specifies the width of the
            search along every longitude. For example, pixel_width=3 will search along the pixel at v=0 and the two
            pixels to its right. The default value will only search along v=0.
        max_accepted_shear: int, optional. Maximum number of pixels to the left of v=0 that are analyzed to look for a
            maximum.

        Returns
        -------
        list: data of every detected shear. Each element has 0: bounds of the detected shear in pixels, 1: shear width
            in km/s, 2: coordinates of the max shear point.
        """
        shear_bounds = self.get_point_groups(
            self.get_shear_points(tolerance, pixel_width), max_regroup_separation)
        if shear_bounds is not None:
            shear_data = []
            for bounds in shear_bounds:
                shear_width = self.get_max_shear_width(bounds, max_accepted_shear, accepted_width)
                if shear_width[0] is not None:
                    shear_data.append((bounds, *shear_width))
            return shear_data

    def get_shear_points(self, tolerance: float, pixel_width: int=1) -> list:
        """
        Get every signal drop along a line from v=0 to the pixel width.

        Arguments
        ---------
        tolerance: float. Controls the sensitivity of signal drop detection. The value given corresponds to the
            percentage of the average peak value along each horizontal line (determined with the
            get_horizontal_maximums method) that will be considered as a signal drop. E.g. for a value of 0.5, if a
            pixel along the central line has a value beneath 0.5 times the average peak value, it will be flagged as a
            potential signal drop.
        pixel_width: int, default=1. This parameter must be greater than or equal to 1 and specifies the width of the
            search along every longitude. For example, pixel_width=3 will search along the pixel at v=0 and the two
            pixels to its right. The default value will only search along v=0.

        Returns
        -------
        list: y_value of every detected shear.
        """
        # Extract useful informations
        maxs = self.get_horizontal_maximums()
        y_slice = slice(*self.y_limits)
        average_max = np.mean(maxs[y_slice,1])

        # Get the array representing every y value that has a drop
        y_shear = np.where(np.any(self.data[:,self.x_center:self.x_center+pixel_width] <= average_max * tolerance, 
                                  axis=1))[0]
        return y_shear[(self.y_limits[0] <= y_shear) & (y_shear <= self.y_limits[1])]

    def get_max_shear_width(self, bounds: list, max_accepted_shear: int, accepted_width: int=None) -> tuple:
        """
        Get the maximum shear width of a certain region by computing the maximum distance between v=0 and the intensity
        peaks in a certain region in addition to its coordinates.

        Arguments
        ---------
        bounds: list. Values between which the search for the farthest maximum should be made.
        max_accepted_shear: int. Maximum number of pixels to the left of v=0 that are analyzed to look for a maximum.
        accepted_width: int, default=None. Normal width of the HI's central band. Greatest accepted distance, in pixels,
            between the HI's center and the maximum of a horizontal line that will not be flagged as a shear point. For
            example, a value of 0 will remove shears of 0 pixels and a value of 1 will remove shears of 1 pixels. Use
            None to accept all shears.

        Returns
        -------
        tuple: first element is the width in km/s computed using the header's informations and the second element is a
            tuple of the detected maximum's coordinates.
        """
        maxs = self.get_horizontal_maximums()[slice(bounds[0], bounds[1]+1),0]
        if max_accepted_shear:
            np.place(maxs, maxs > max_accepted_shear + self.x_center, 0)

        if not np.all(maxs) == 0:
            # Compute the distance using the header and convert m/s to km/s
            max_width = - (np.max(maxs - self.x_center))
            if accepted_width is not None:
                if np.abs(max_width) <= accepted_width:
                    return None, None
            coords = np.max(maxs), np.argmax(maxs) + bounds[0]
            return max_width * self.header["CDELT1"] / 1000, coords

        else:
            return None, None


class Spider_slice(HI_slice):
    """ 
    Encapsulate the methods specific to the analysis of slices of Spider data cubes.
    """

    def plot(self, bounds: list=None, shear_width: float=None, max_coords: tuple=None, fullscreen: bool=False):
        """
        Plot the object in a two-dimensional plot.

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
        if bounds is not None and shear_width is not None and max_coords is not None:
            x_center = max_coords[0] - shear_width/(self.header["CDELT1"]/1000)
            min_xlim = min(x_center - 10, max_coords[0] - 10)
            max_xlim = max(x_center + 10, max_coords[0] + 10)
            plt.xlim(min_xlim, max_xlim)
            plt.ylim(bounds[0]-40, bounds[1]+40)

            # Build the rectangle around the detected shear and enlarge so that the border is at the outside of the
            # concerned pixels
            region = mpl.patches.Rectangle(
                (min(x_center, max_coords[0]) - 0.5, min(bounds[0], max_coords[1]) - 0.5), 
                np.abs(max_coords[0] - x_center) + 1, np.abs(bounds[1] - bounds[0]) + 1,
                linewidth=1, edgecolor='r', facecolor='none'
            )
            # Build the rectangle around the max shear detected
            max_shear = mpl.patches.Rectangle(
                (max_coords[0] - 0.5, max_coords[1] - 0.5), 1, 1, linewidth=1, edgecolor='w', facecolor="none"
            )
            ax.add_patch(region)
            ax.add_patch(max_shear)
            z = self.z_coordinate

            # Set info parameters
            if "GLAT" in self.header["CTYPE3"]:
                plt.title(
                    f"Current z_coordinate: {z} ({b.from_pixel(z, self.header)}), shear_width: {shear_width:.2f} km/s"
                )
            else:
                plt.title(
                    f"Current z_coordinate: {z}, shear_width: {shear_width:.2f} km/s"
                )
        
        if fullscreen:
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
        plt.show()

    def check_shear(self, *,
            rejection: float,
            accepted_width: int,
            max_regroup_separation: int=0,
            max_accepted_shear: int=None
        ) -> list:
        """
        Extract potential shear zones and their maximum shear value in km/s.

        Arguments
        ---------
        rejection: float. Controls the sensitivity of the shear detection. The value given corresponds to the minimum
            of the percentage of the average peak value along each horizontal line (determined with the
            get_horizontal_maximums method) that will be considered as a HI shift. E.g. for a value of 0.5, if a
            sufficiently displaced maximum has a value beneath 0.5 times the average peak value, it will be considered
            as too faint for a shear point and will be disregarded.
        accepted_width: int. Normal width of the HI's central band. Greatest accepted distance, in pixels, between the
            HI's center and the maximum of a horizontal line that will not be flagged as a shear point.
        max_regroup_separation: int, default=0. Maximum separation of two consecutive points that will be considered to
            belong to the same signal drop section. This controls how many different regions will be outputted and will
            merge those that are close.
        max_accepted_shear: int, optional. Maximum number of pixels to the left or right of the central band that are
            analyzed to look for a maximum.

        Returns
        -------
        list: data of every detected shear. Each element has 0: bounds of the detected shear in pixels, 1: shear width
            in km/s, 2: coordinates of the max shear point.
        """
        shear_bounds = self.get_point_groups(self.get_shear_points(rejection, accepted_width), max_regroup_separation)
        if shear_bounds is not None:
            shear_data = []
            for bounds in shear_bounds:
                shear_width = self.get_max_shear_width(bounds, max_accepted_shear)
                if shear_width[0] is not None:
                    shear_data.append((bounds, *shear_width))
            return shear_data
    
    def get_shear_points(self, rejection: float, accepted_width: int) -> list:
        """
        Get every shear whose intensity is greater to a certain number controlled by the rejection float provided.

        Arguments
        ---------
        rejection: float. Controls the sensitivity of the shear detection. The value given corresponds to the minimum
            of the percentage of the average peak value along each horizontal line (determined with the
            get_horizontal_maximums method) that will be considered as a HI shift. E.g. for a value of 0.5, if a
            sufficiently displaced maximum has a value beneath 0.5 times the average peak value, it will be considered
            as too faint for a shear point and will be disregarded.
        accepted_width: int. Greatest accepted distance, in pixels, between the HI's center and the maximum of a
            horizontal line that will not be flagged as a shear point.

        Returns
        -------
        list: y_value of every detected shear.
        """
        # Extract useful informations
        maxs = self.get_horizontal_maximums()
        y_slice = slice(*self.y_limits)
        maxs_values = maxs[y_slice,1]
        average_max = np.mean(maxs_values[maxs_values > np.median(maxs_values)])
        x_center = scipy.stats.mode(maxs[y_slice,0])[0]

        # Get the array representing every y value that has a sufficiently displaced maximum and whose maximum is
        # bright enough
        y_shear = np.where((np.abs(maxs[:,0] - x_center) > accepted_width) & 
                           (maxs[:,1] >= rejection * average_max))[0]
        # Return only the values between the y_limits
        return y_shear[(self.y_limits[0] <= y_shear) & (y_shear <= self.y_limits[1])]
    
    def get_max_shear_width(self, bounds: list, max_accepted_shear: int) -> tuple:
        """
        Get the maximum shear width of a certain region by computing the maximum distance between the most frequent
        peak value and the intensity peaks in a certain region in addition to its coordinates.

        Arguments
        ---------
        bounds: list. Values between which the search for the farthest maximum should be made.
        max_accepted_shear: int. Maximum number of pixels to the left or right of the most frequent speed that are
            analyzed to look for a maximum.

        Returns
        -------
        tuple: first element is the width in km/s computed using the header's informations and the second element is a
            tuple of the detected maximum's coordinates.
        """
        maxs = self.get_horizontal_maximums()[slice(bounds[0], bounds[1]+1),0]
        x_center = float(scipy.stats.mode(self.get_horizontal_maximums()[slice(*self.y_limits),0])[0])

        if max_accepted_shear:
            np.place(maxs, np.abs(maxs - x_center) > max_accepted_shear, 0)
        
        if not np.all(maxs) == 0:
            # Find the relative horizontal position of the maximum shear
            max_rel_y_pos = np.argmax(np.abs(maxs - x_center))
            # Compute the distance using the header and convert m/s to km/s
            max_width = (maxs[max_rel_y_pos] - x_center) * self.header["CDELT1"] / 1000
            coords = maxs[max_rel_y_pos], max_rel_y_pos + bounds[0]
            return max_width, coords
        else:
            return None, None


class LOOP4_cube(HI_cube):
    slice_type = LOOP4_slice


class Spider_cube(HI_cube):
    slice_type = Spider_slice

