from __future__ import annotations
import numpy as np
from astropy.io import fits
from copy import deepcopy

from src.hdu.fits_file import FitsFile
from src.hdu.maps.map import Map
from src.hdu.arrays.array_2d import Array2D
from src.hdu.arrays.array_3d import Array3D
from src.headers.header import Header


class GroupedMaps(FitsFile):
    """
    Encapsulates the necessary methods to compare and linked maps.
    """

    def __init__(self, maps: list[tuple[str, list[Map]]]):
        """
        Initializes a GroupedMaps object.

        Parameters
        ----------
        maps : list[tuple[str, list[Map]]]
            List of maps that are linked together. The first element is the map's name as it will be appear in the
            header or accessed by attribute and the second element is the list of Maps that represent the same quantity.
        """
        self.names = []
        for name, map_list in maps:
            setattr(self, name, map_list)
            self.names.append(name)

    def __getitem__(self, name: str) -> list[Map]:
        return self.names

    @classmethod
    def load(cls, filename: str) -> GroupedMaps:
        """
        Loads a GroupedMaps from a .fits file.

        Parameters
        ----------
        filename : str
            Name of the file to load.

        Returns
        -------
        GroupedMaps : GroupedMaps
            Loaded GroupedMaps.
        """
        hdu_list = fits.open(filename)
        new_header = hdu_list[0].header.copy()
        for key in deepcopy(list(new_header.keys())):
            if key.startswith("IMAGE") or key.startswith("EXT") or key == "NAXIS3":
                del new_header[key]
        new_header["NAXIS"] = 2

        maps = []
        header_frames = [hdu_list[0].header[f"IMAGE{i}"] for i in range(1, hdu_list[0].shape[0] + 1)]
        unique_names = list(dict.fromkeys(header_frames))
        identical_frames = len(header_frames) // len(unique_names)

        # The same number of images is present for every key
        for i, name in enumerate(unique_names):
            current_list = []
            i_offset = i * identical_frames
            for j in range(i_offset, i_offset + identical_frames):
                current_list.append(
                    Map(
                        data=Array2D(hdu_list[0].data[j,:,:]),
                        uncertainties=Array2D(hdu_list[1].data[j,:,:]) if len(hdu_list) > 1 else np.NAN,
                        header=Header(new_header),
                    )
                )
            maps.append((deepcopy(name), current_list))

        gm = cls(maps)
        return gm

    def save(self, filename: str, overwrite: bool=False):
        """
        Saves a GroupedMaps to a file. Saving a GroupedMaps object is discouraged as the format is more restrictive than
        Tesseracts. This may however be useful for a more descriptive header and when no further change will be made to
        the data.

        Parameters
        ----------
        filename : str
            Filename in which to save the GroupedMaps.
        overwrite : bool, default=False
            Whether the file should be forcefully overwritten if it already exists.
        """
        data = []
        uncertainties = []
        map_occurences = {}     # Tracks the number of maps per name
        
        for name in self.names:
            for map_ in getattr(self, name):
                data.append(map_.data)
                uncertainties.append(map_.uncertainties)
                map_occurences[name] = map_occurences.get(name, 0) + 1
        
        header = getattr(self, self.names[0])[0].header.copy()
        header["EXT0"] = "Data"
        if not np.all(np.isnan(uncertainties)):
            header["EXT1"] = "Uncertainties"
        for i, items in enumerate(map_occurences.items()):
            name, occurences = items
            for j in range(occurences):
                header[f"IMAGE{i*occurences + j + 1}"] = name

        data_array = Array3D(data)
        hdu_list = fits.HDUList([data_array.get_PrimaryHDU(header)])
        if header.get("EXT1") is not None:
            hdu_list.append(Array3D(uncertainties).get_ImageHDU(header))

        super().save(filename, hdu_list, overwrite)
