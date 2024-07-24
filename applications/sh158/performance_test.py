import numpy as np
import graphinglib as gl
from datetime import datetime
import time

from src.tools.statistics.advanced_stats import structure_function, autocorrelation_function, \
                                                autocorrelation_function_2d, increments
from src.hdu.maps.map import Map

test = Map.load("summer_2023/gaussian_fitting/maps/external_maps/"
                      +"dens_it_sii_sans_fcorr_nii_plus_plusmin_pouss_seuil_errt_1000.fits")

with open("applications/sh158/performances.txt", "a") as f:
    f.write(f"{datetime.now().strftime("%H:%M:%S")}\n")
    start_total = time.time()
    start = time.time()
    data = autocorrelation_function(test.data)
    f.write(f"autocorrelation_function :    {time.time() - start:.5f}s\n")
    start = time.time()
    data = autocorrelation_function_2d(test.data)
    f.write(f"autocorrelation_function_2d : {time.time() - start:.5f}s\n")
    start = time.time()
    data = structure_function(test.data)
    f.write(f"structure_function :          {time.time() - start:.5f}s\n")
    start = time.time()
    data = increments(test.data)
    f.write(f"increment :                   {time.time() - start:.5f}s\n")
    f.write(f"                 Total time : {time.time() - start_total:.5f}s\n")
    f.write("\n\n")
