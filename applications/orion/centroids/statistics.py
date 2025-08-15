"""
This file contains statistics code that is ran on Hypatia.
"""

import numpy as np
import scipy as sp
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
from src.tools.statistics.advanced_stats import increments


# OPTIMAL WIDTH
# -------------
# Info = namedtuple("Info", ["name", "cube_filename", "gaussian_index", "relative_error_threshold"])

# files = [
#     Info("nii_1", "nii_1_binned_3x3.fits", 1, 0.0016),
#     Info("nii_2", "nii_2_binned_4x4.fits", 1, 0.003),
#     Info("oiii_1", "oiii_1_binned_3x3.fits", 0, 0.09),
#     Info("oiii_2", "oiii_2_binned_4x4.fits", 0, 0.03),
#     Info("sii_1", "sii_1_binned_3x3.fits", [0, 1], 0.009),
#     Info("sii_2", "sii_2_binned_4x4.fits", [0, 1], 0.016),
#     Info("ha_1", "ha_1_binned_3x3.fits", "variable", 0.015),
#     Info("ha_2", "ha_2_binned_4x4.fits", 0, 0.005),
# ]


# width_range = np.arange(3, 101, 2)
# open_write = lambda text: open("applications/orion/centroids/zurflueh_widths.txt", "a").write(text)

# for info in files:
#     centroids = GroupedMaps.load(f"data/orion/centroids/maps/{info.name}_centroids.fits").centroids
#     centroid_data = np.stack([centroid.data for centroid in centroids], axis=0)
#     for component_i, data in enumerate(centroid_data):
#         header = f"{info.name}, component {component_i}\n"
#         open_write(f"{header}{"=" * len(header)}\n")
#         for width in smart_tqdm(width_range, colour="RED", desc=info.name):
#             gradient = zfilter(data, width)
#             filtered_map = data - gradient
#             res = evaluate_delta_f2(filtered_map)
#             if res:
#                 open_write(f"width={width:03}, ∆F_2(tau_0)={res}\n")
#             else:
#                 open_write(f"width={width:03}, INVALID\n")
#         open_write("\n")
#     open_write("\n")


optimal_widths_dict = {
    "nii_1, component 0": 33,
    "nii_2, component 0": 41,
    "oiii_1, component 0": 93,
    "oiii_2, component 0": 33,
    "sii_1, component 0": 39,
    "sii_1, component 1": 45,
    "sii_2, component 0": 43,
    "sii_2, component 1": 47,
    "ha_1, component 0": 57,
    "ha_1, component 1": 31,
    "ha_2, component 0": 31,
}

load = lambda field: GroupedMaps.load(f"data/orion/centroids/maps/{field}_centroids.fits").centroids

centroid_maps = {
    "[NII]": [
        [load(f"nii_{i}")[0], optimal_widths_dict[f"nii_{i}, component 0"]] for i in [1, 2]
    ],
    "[OIII]": [
        [load(f"oiii_{i}")[0], optimal_widths_dict[f"oiii_{i}, component 0"]] for i in [1, 2]
    ],
    "[SII]": [
        [load("sii_1")[0], optimal_widths_dict["sii_1, component 0"]],
        [load("sii_2")[0], optimal_widths_dict["sii_2, component 0"]],
        [load("sii_1")[1], optimal_widths_dict["sii_1, component 1"]],
        [load("sii_2")[1], optimal_widths_dict["sii_2, component 1"]],
    ],
    r"H$\alpha$": [
        [load("ha_1")[0], optimal_widths_dict["ha_1, component 0"]],
        [load("ha_2")[0], optimal_widths_dict["ha_2, component 0"]],
        [load("ha_1")[1], optimal_widths_dict["ha_1, component 1"]],
    ],
}
all_filtered_centroids = {
    key: [centroid.data - zfilter(centroid.data, width) for centroid, width in centroids]
    for key, centroids in centroid_maps.items()
}

# INCREMENTS
# ----------
# def get_increment_plottables(data: np.ndarray) -> list:
#     sorted_data = np.sort(np.concatenate((data, -data)))

#     bin_width = 0.1
#     max_bin = np.nanmax(sorted_data + bin_width)//bin_width * bin_width
#     bins = np.arange(-max_bin, max_bin, bin_width)

#     hist = gl.Histogram(
#         sorted_data,
#         number_of_bins=bins,
#         show_params=False,
#     )
#     amplitude = np.max(hist.bin_heights)

#     stddev = hist.bin_centers[np.argmin(np.abs(hist.bin_heights - np.max(hist.bin_heights)/2))] / (np.sqrt(2*np.log(2)))

#     curve = gl.Curve.from_function(
#         func=lambda x: amplitude * np.exp(-x**2/(2*stddev**2)),
#         x_min=-5,
#         x_max=5,
#         color="black",
#         line_width=1.4,
#     )

#     return [hist, curve]

# all_figs = []
# BIN_TOLERANCE = 0.1
# increment_values = [2, 4, 6, 10]
# FIELD = 2
# for field, filtered_centroids in smart_tqdm(all_filtered_centroids.items()):
# # for field, filtered_centroids in all_filtered_centroids.items():
#     # increments_data = [increments(Map(centroid).crop_nans().data) for centroid in filtered_centroids]
#     increments_data = [increments(Map(filtered_centroids[FIELD - 1]).crop_nans().data)]
#     for increment in increment_values:
#         current_increment_data = []
#         for component in increments_data:  # increments_data is a list of 2d arrays for each component
#             for increment_i, values in component.items():
#                 if increment + BIN_TOLERANCE > increment_i > increment - BIN_TOLERANCE:
#                     current_increment_data.extend(values.tolist())
#         current_increment_data = np.array(current_increment_data)

#         fig = gl.SmartFigure(
#             x_label=rf"$\gamma_2={sp.stats.kurtosis(current_increment_data)}$",
#             # x_label=rf"$\gamma_2=BLABLAC$",
#             x_lim=(-2.5, 2.5),
#             elements=get_increment_plottables(current_increment_data),
#             # elements=[gl.Arrow((0,0), (1,1))],
#             reference_labels=False,
#             global_reference_label=(field == "[NII]")
#         )
#         all_figs.append(fig)

# increments_fig = gl.SmartFigure(
#     4,
#     4,
#     x_label="Normalized increment [-]",
#     y_label="Normalized count [-]",
#     elements=all_figs,
#     size=(13, 9),
#     # subtitles=[r"$\Delta=2$", r"$\Delta=4$", r"$\Delta=6$", r"$\Delta=10$"] + [None] * 12,
#     subtitles=[rf"$\Delta={val}$" for val in increment_values] + [None] * 12,
#     height_padding=0.05,
#     height_ratios=[1.18, 1, 1, 1],
# )
# for i, field in enumerate(centroid_maps.keys()):
#     increments_fig[i, 1].add_elements([gl.Text(1.07, 1.04, field, relative_to="figure", font_size=14)])

# increments_fig.save(f"figures/orion/centroids/increments_histograms_field_{FIELD}.pdf", dpi=600)


# STRUCTURE FUNCTIONS
# -------------------
all_figs = []
FIELD = 2
# fit_lengths = [(0.45, 1.15), (0.45, 1.1), (0.45, 0.92), (0.45, 0.92)]
for field, filtered_centroids in smart_tqdm(all_filtered_centroids.items()):
# for field, filtered_centroids in all_filtered_centroids.items():
    # structure_func_data = [structure_function(Map(centroid).crop_nans().data) for centroid in filtered_centroids]
    structure_func_data = [structure_function(Map(filtered_centroids[FIELD - 1]).crop_nans().data)]
    fig = get_fitted_structure_function_figure(structure_func_data[0], (0.45, 1), 10000)
    fig.subtitles = [field]
    all_figs.append(fig)

structure_func_fig = gl.SmartFigure(
    2,
    2,
    x_label="Lag [pixels]",
    y_label="Structure function [-]",
    elements=all_figs,
    size=(13, 9),
)
structure_func_fig.save(f"figures/orion/linewidths/structure_functions_field_{FIELD}.pdf", dpi=600)
