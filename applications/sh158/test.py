import numpy as np
import graphinglib as gl
import pyregion
from scipy.optimize import curve_fit
import dill
from functools import partial
import time

from src.statistics.advanced_stats import autocorrelation_function, autocorrelation_function_2d, test
from src.hdu.maps.map import Map

turbulence = Map.load("summer_2023/gaussian_fitting/maps/computed_data_selective/turbulence.fits")

regions = [
    ("Global region", None, 50),
    ("Diffuse region", pyregion.open("summer_2023/gaussian_fitting/regions/region_1.reg"), 20),
    ("Central region", pyregion.open("summer_2023/gaussian_fitting/regions/region_2.reg"), 10),
    ("Filament region", pyregion.open("summer_2023/gaussian_fitting/regions/region_3.reg"), 6)
]

start = time.time()
try:
    with open("applications/sh158/saved_data/2d_acr_func_turbulence_global.gz", "rb") as f:
        data = dill.load(f)
except:
    data = autocorrelation_function_2d(turbulence.data)
    with open("applications/sh158/saved_data/2d_acr_func_turbulence_global.gz", "wb") as f:
        dill.dump(data, f)

print("Time :", time.time() - start)

with open("applications/sh158/saved_data/2d_acr_func_turbulence_global_.gz", "rb") as f:
    ref_data = dill.load(f)

if np.allclose(data, ref_data):
    print("Success")
else:
    print("Failure")
