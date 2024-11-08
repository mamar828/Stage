import graphinglib as gl
from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
import pyregion
import dill
from functools import partial
import time
from eztcolors import Colors as C
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.tools.zurflueh_filter.zfilter import zfilter, create_zfilter
from src.tools.statistics.advanced_stats import *
from src.hdu.maps.map import Map
from src.hdu.arrays.array_2d import Array2D
from src.hdu.cubes.cube import Cube
from src.tools.mask import Mask
from src.hdu.maps.convenient_funcs import get_speed


def power_function(x, a, m):
    return a * x**m

def estimate_params(data, number_of_iterations: int=1000):
    sample_data = np.array([np.random.normal(mu, abs(sigma), number_of_iterations) for mu, sigma in data[:,1:3]])
    parameters = []
    for i in range(number_of_iterations):
        parameters.append(curve_fit(
            f=power_function,
            xdata=data[:,0],
            ydata=sample_data[:,i],
            p0=[0.3, 0.5],
            maxfev=100000,
            bounds=(0.05, 2))[0]
        )
    
    parameters_array = np.array(parameters)
    results = {
        "a": np.array([np.mean(parameters_array[:,0]), np.std(parameters_array[:,0])]),
        "m": np.array([np.mean(parameters_array[:,1]), np.std(parameters_array[:,1])]),
    }
    return results

def get_plottables(data: np.ndarray, fit_length: float) -> list:
    sorted_data = data[np.argsort(data[:,0])]
    scatter = gl.Scatter(
        sorted_data[:,0],
        sorted_data[:,1],
        marker_size=3,
        face_color="black",
    )

    scatter.add_errorbars(
        y_error=np.abs(sorted_data[:,2]),
        errorbars_line_width=0.25,
        cap_width=0,
    )

    mask = (sorted_data[:,0] <= fit_length)
    pm = estimate_params(sorted_data[mask], 100)
    fit = gl.Curve.from_function(
        func=partial(power_function, a=pm["a"][0], m=pm["m"][0]),
        x_min=0,
        x_max=sorted_data[np.argmax(sorted_data[mask,1]),0],        # fit until the structure function's first peak
        label=rf"Slope : {pm["m"][0]:.3f} ± {pm["m"][1]:.3f}",
        line_width=1,
        color="red",
    )

    return [scatter, fit]

def get_log_plottables(data: np.ndarray, fit_length: float) -> list:
    sorted_data = np.log10(data[np.argsort(data[:,0]),:2])
    scatter = gl.Scatter(
        sorted_data[:,0],
        sorted_data[:,1],
        marker_size=3,
        face_color="black",
    )

    mask = (0.45 <= sorted_data[:,0] <= fit_length)
    fit = gl.FitFromPolynomial(
        scatter.create_slice_x(0, sorted_data[np.argmax(sorted_data[mask,1]),0]),
        1,
        line_width=1,
        color="red",
    )

    return [scatter, fit]

target_region = pyregion.open("data/sh158/target_region.reg")       # this region contains all the object
regions = [
    # ("Sh2-158", None, 50),
    ("Sh2-158", pyregion.open("data/sh158/turb.reg"), 30),
    ("Diffuse region", pyregion.open("summer_2023/gaussian_fitting/regions/region_1.reg"), 20),
    ("Central region", pyregion.open("summer_2023/gaussian_fitting/regions/region_2.reg"), 10),
    ("Filament region", pyregion.open("summer_2023/gaussian_fitting/regions/region_3.reg"), 10)
]

def generate_figure(data: Map, figure_filename: str, scale: str="linear"):
    figs = []
    for name, region, fit_length in regions:
        data_masked = data.get_masked_region(region)
        # fig = gl.Figure(); fig.add_elements(data_masked.data.plot); fig.show()
        str_data = structure_function(data_masked.data)

        if scale == "linear":
            fig = gl.Figure(title=name, x_lim=(0, fit_length*1.2))
            plottables = get_plottables(str_data, fit_length)
            cropped_scatter = plottables[0].create_slice_x(0, fit_length*1.2)
            fig.y_lim = np.min(cropped_scatter.y_data)-0.2, np.max(cropped_scatter.y_data)+0.3
        elif scale == "log":
            fig = gl.Figure(title=name, x_lim=(0, np.log10(fit_length*1.2)))
            plottables = get_log_plottables(str_data, np.log10(fit_length))
            cropped_scatter = plottables[0].create_slice_x(0, np.log10(fit_length*1.2))
            fig.y_lim = -0.2, 0.5
        fig.add_elements(*plottables)
        figs.append(fig)

    multifig = gl.MultiFigure.from_grid(figs, (2,2), (13, 8.6))
    multifig.x_label = "Lag [pixels]"
    multifig.y_label = "Structure Function [-]"
    multifig.save(figure_filename)


# Turbulence structure function ok
# -----------------------------
# generate_figure(
#     Map.load("summer_2023/gaussian_fitting/maps/computed_data_selective/turbulence.fits"),
#     "figures/sh158/advanced_stats/structure_function/test.pdf"
# )

# NII mean test
# -------------
# generate_figure(
#     Map.load("data/sh158/fit_no_bin/NII_mean.fits"),
#     "figures/sh158/nii_mean/no_zurflueh.pdf"
# )
# Try with conversion to km/s
# centroids = Map.load("data/sh158/fit_no_bin/NII_mean.fits")
# cube = Cube.load("summer_2023/gaussian_fitting/data_cubes/night_34_wcs.fits")
# speed = get_speed(centroids, cube)
# generate_figure(
#     speed,
#     "figures/sh158/nii_mean/no_zurflueh_speed.pdf"
# )

# Zurflueh width test
# -------------------
# m = Map.load("data/sh158/fit_no_bin/NII_mean.fits").get_masked_region(target_region)
# for width in tqdm([81], colour="RED"):
#     gradient = zfilter(m.data, width)
#     fig = gl.Figure(); fig.add_elements(gl.Heatmap(m.data - gradient, origin_position="lower", vmin=-1, vmax=1)); fig.show()
#     fig = gl.Figure(); fig.add_elements(gl.Heatmap(gradient, origin_position="lower")); fig.show()
#     filtered_map = Map(m.data - gradient, m.uncertainties, m.header)
#     # fig = gl.Figure(); fig.add_elements(filtered_map.data.plot); fig.show()
#     generate_figure(
#         filtered_map,
#         f"figures/sh158/nii_mean/log/zurflueh_log_w={width}.pdf",
#         "log"
#     )

# Random array testing
# --------------------
# rand = np.random.normal(0, 1, (100,100))
# data = rand + np.tile(np.linspace(3,7,100), (100,1))
# data[60:,70:] += np.tile(np.linspace(0,5,30), (40,1))
# figs = [gl.Figure() for _ in range(3)]
# figs[0].add_elements(gl.Heatmap(data, origin_position="lower"))
# grad = zfilter(data, 33)
# figs[1].add_elements(gl.Heatmap(grad, origin_position="lower"))
# figs[2].add_elements(gl.Heatmap(data - grad, origin_position="lower"))
# multifig = gl.MultiFigure.from_row(figs); multifig.show()

# ACR functions
# -------------
# m = Map.load("data/sh158/fit_no_bin/NII_mean.fits").get_masked_region(pyregion.open("data/sh158/turb.reg"))
# for width in tqdm([21,27,31,37,41,47,51,57,61,67,71,77,81,87,91,97,101], colour="RED"):
#     gradient = zfilter(m.data, width)
#     # fig = gl.Figure(); fig.add_elements(gl.Heatmap(gradient, origin_position="lower")); fig.show()
#     filtered_map = Map(m.data - gradient, m.uncertainties, m.header)
#     autocorrelation_1d = autocorrelation_function(filtered_map.data, "Boily")
#     autocorrelation_1d_curve = gl.Curve(autocorrelation_1d[:,0], autocorrelation_1d[:,1])
#     fig = gl.Figure()
#     fig.add_elements(autocorrelation_1d_curve)
#     fig.save(f"figures/sh158/nii_mean/acr_func/zurflueh_w={width}.pdf")

# ∆F_2_tau_0
# ----------
# m = Map.load("data/sh158/fit_no_bin/NII_mean.fits").get_masked_region(pyregion.open("data/sh158/turb.reg"))
# for width in tqdm([21,27,31,37,41,47,51,57,61,67,71,77,81,87,91,97,101], colour="RED"):
#     gradient = zfilter(m.data, width)
#     # fig = gl.Figure(); fig.add_elements(gl.Heatmap(gradient, origin_position="lower")); fig.show()
#     filtered_map = Map(m.data - gradient, m.uncertainties, m.header)
#     res, fig = evaluate_delta_f2(filtered_map.data)
#     fig.save(f"figures/sh158/nii_mean/acr_func/zurflueh_w={width}_str_fit.pdf")
#     with open("figures/sh158/nii_mean/acr_func/results.txt", "a") as f: 
#         if res:
#             f.write(f"width={width:03}, ∆F_2(tau_0)={res}\n")
#         else:
#             f.write(f"width={width:03}, INVALID\n")

# Structure function figure
# -------------------------
m = Map.load("data/sh158/fit_no_bin/NII_mean.fits").get_masked_region(target_region)
gradient = zfilter(m.data, 81)
filtered_map = Map(m.data - gradient, m.uncertainties, m.header)
fit_lengths = [(0.45, 1.1)]*4
# fit_lengths = [(0.45, 1.15), (0.45, 1.2), (0.45, 1), (0.45, 1.05)]
figs = []
for (name, region, _), fit_length in zip(regions, fit_lengths):
    if False:
        data_masked = filtered_map.get_masked_region(region)
        str_data = structure_function(data_masked.data)
        np.save(f"figures/sh158/nii_mean/{name}.npy", str_data)
        figs.append(get_fitted_structure_function_figure(str_data, fit_length, 10000))
    else:
        str_data = np.load(f"figures/sh158/nii_mean/{name}.npy")
        fig = get_fitted_structure_function_figure(str_data, fit_length, 10000)
        fig.title = name
        figs.append(fig)
figs[0].y_lim = 0.1, 0.4
figs[1].y_lim = 0, 0.4
figs[2].y_lim = -0.3, 0.45
figs[3].y_lim = -0.3, 0.45
multifig = gl.MultiFigure.from_grid(figs, (2,2), (13, 8.6))
multifig.x_label = "Lag [pixels]"
multifig.y_label = "Structure Function [-]"
# multifig.show()
multifig.save("figures/sh158/nii_mean/str_func_same_lengths.pdf", dpi=600)

# Individual regions
# ------------------
# m = Map.load("data/sh158/fit_no_bin/NII_mean.fits")
# width_ranges = [range(53,85,2), range(13,45,2), range(13,45,2)]
# # range(3,113,10)
# with open("applications/sh158/results.txt", "a") as f: 
#     for (name, region, _), width_range in zip(regions[1:], width_ranges):
#         masked = m.get_masked_region(region)
#         f.write(f"{name}\n")
#         for width in tqdm(width_range, colour="RED", desc=name):
#             gradient = zfilter(masked.data, width)
#             filtered_map = Map(masked.data - gradient, masked.uncertainties, masked.header)
#             res = evaluate_delta_f2(filtered_map.data)
#             if res:
#                 f.write(f"width={width:03}, ∆F_2(tau_0)={res}\n")
#             else:
#                 f.write(f"width={width:03}, INVALID\n")

# Individual regions structure function figures
# ---------------------------------------------
# m = Map.load("data/sh158/fit_no_bin/NII_mean.fits")
# zfilter_widths = [81, 65, 31, 33]
# fit_lengths = [(0.45, 1.15), (0.45, 1.1), (0.45, 0.9), (0.45, 0.9)]
# figs = []
# for (name, region, _), width, fit_length in zip(regions, zfilter_widths, fit_lengths):
#     masked = m.get_masked_region(region)
#     gradient = zfilter(masked.data, width)
#     filtered_map = Map(m.data - gradient, m.uncertainties, m.header)
#     if name == "Sh2-158":
#         str_data = np.load(f"figures/sh158/nii_mean/{name}.npy")
#     else:
#         str_data = structure_function(filtered_map.data)
#     fig = get_fitted_structure_function_figure(str_data, fit_length, 10000)
#     fig.title = f"{name} $w={width}$"
#     figs.append(fig)

# figs[0].y_lim = 0.1, 0.35
# figs[1].y_lim = 0.1, 0.35
# figs[2].y_lim = 0, 0.45
# figs[3].y_lim = 0, 0.45
# multifig = gl.MultiFigure.from_grid(figs, (2,2), (13, 8.6))
# multifig.x_label = "Lag [pixels]"
# multifig.y_label = "Structure Function [-]"
# # multifig.show()
# multifig.save("figures/sh158/nii_mean/str_func_variable_zfilter.pdf", dpi=600)

# Autocorrelation function figures
# --------------------------------
# m = Map.load("data/sh158/fit_no_bin/NII_mean.fits")
# zfilter_widths = [81, 65, 31, 33]

# # Test for same zfilter, must comment out the second zfilter calculation in the loop
# masked = m.get_masked_region(target_region)
# gradient = zfilter(masked.data, 81)
# fmap = Map(m.data - gradient, m.uncertainties, m.header)

# figs = {"1D" : [], "2D" : []}
# for (name, region, _), width in tqdm(zip(regions, zfilter_widths)):
#     # masked = m.get_masked_region(region)
#     # gradient = zfilter(masked.data, width)
#     # filtered_map = Map(m.data - gradient, m.uncertainties, m.header)
#     filtered_map = fmap.get_masked_region(region)

#     for key, func, plot_func in zip(
#         ["1D", "2D"],
#         [autocorrelation_function, autocorrelation_function_2d],
#         [get_autocorrelation_function_scatter, get_autocorrelation_function_2d_contour]
#     ):
#         figs[key].append(gl.Figure(title=f"{name}"))# $w={width}$"))
#         figs[key][-1].add_elements(plot_func(func(filtered_map.data)))

# for fig, x_upper_bound in zip(figs["1D"], [40,30,25,20]):
#     fig.x_lim, fig.y_lim, fig.show_grid = (0, x_upper_bound), (-0.3, 0.6), True
# multifig_1d = gl.MultiFigure.from_grid(figs["1D"], (2,2), (13, 8.6))
# multifig_1d.x_label = "Lag [pixels]"
# multifig_1d.y_label = "Autocorrelation Function [-]"
# multifig_1d.save("figures/sh158/nii_mean/acr_func_1d_same_zfilter.pdf", dpi=600)

# multifig_2d = gl.MultiFigure.from_grid(figs["2D"], (2,2), (13, 8.6))
# multifig_2d.x_label = "x lag [pixels]"
# multifig_2d.y_label = "y lag [pixels]"
# multifig_2d.save("figures/sh158/nii_mean/acr_func_2d_same_zfilter.pdf", dpi=600)
