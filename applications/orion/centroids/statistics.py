import numpy as np
import src.graphinglib as gl
import pyregion
from collections import namedtuple
from tqdm import tqdm

from src.hdu.tesseract import Tesseract
from src.hdu.cubes.cube import Cube
from src.hdu.maps.map import Map
from src.hdu.maps.grouped_maps import GroupedMaps
from src.tools.messaging import smart_tqdm
from src.tools.zurflueh_filter.zfilter import zfilter
from src.tools.statistics.advanced_stats import *


Info = namedtuple("Info", ["name", "cube_filename", "gaussian_index", "relative_error_threshold"])

files = [
    Info("nii_1", "nii_1_binned_3x3.fits", 1, 0.0016),
    Info("nii_2", "nii_2_binned_4x4.fits", 1, 0.003),
    Info("oiii_1", "oiii_1_binned_3x3.fits", 0, 0.09),
    Info("oiii_2", "oiii_2_binned_4x4.fits", 0, 0.03),
    Info("sii_1", "sii_1_binned_3x3.fits", [0, 1], 0.009),
    Info("sii_2", "sii_2_binned_4x4.fits", [0, 1], 0.016),
    Info("ha_1", "ha_1_binned_3x3.fits", "variable", 0.015),
    Info("ha_2", "ha_2_binned_4x4.fits", 0, 0.005),
]


# OPTIMAL WIDTH
# -------------
width_range = np.arange(3, 101, 2)
open_write = lambda text: open("applications/orion/centroids/zurflueh_widths.txt", "a").write(text)

for info in files:
    centroids = GroupedMaps.load(f"data/orion/centroids/maps/{info.name}_centroids.fits").centroids
    centroid_data = np.stack([centroid.data for centroid in centroids], axis=0)
    for component_i, data in enumerate(centroid_data):
        header = f"{info.name}, component {component_i}\n"
        open_write(f"{header}{"=" * len(header)}\n")
        for width in smart_tqdm(width_range, colour="RED", desc=info.name):
            gradient = zfilter(data, width)
            filtered_map = data - gradient
            res = evaluate_delta_f2(filtered_map)
            if res:
                open_write(f"width={width:03}, ∆F_2(tau_0)={res}\n")
            else:
                open_write(f"width={width:03}, INVALID\n")
        open_write("\n")
    open_write("\n")
