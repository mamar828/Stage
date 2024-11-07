import numpy as np
import graphinglib as gl
from datetime import datetime
import time

from src.tools.statistics.advanced_stats import structure_function, autocorrelation_function, \
                                                autocorrelation_function_2d, increments
from src.hdu.maps.map import Map

test = Map.load("summer_2023/gaussian_fitting/maps/external_maps/"
               +"dens_it_sii_sans_fcorr_nii_plus_plusmin_pouss_seuil_errt_1000.fits")

# fig1 = gl.Figure()
# fig1.add_elements(test.data.plot)
# fig2 = gl.Figure()
# fig2.add_elements(test.crop_nans().bin((4,4),True).data.plot)
# multifig = gl.MultiFigure.from_row([fig1, fig2])
# multifig.show()
# raise

with open("applications/sh158/acr_performance/info.txt", "a") as f:
    f.write(f"{datetime.now().strftime("%H:%M:%S")}\n")
    start = time.time()
    data = autocorrelation_function(test.crop_nans().data)
    fig = gl.Figure(); fig.add_elements(gl.Curve(data[:,0], data[:,1])); fig.save("applications/sh158/acr_performance/a.pdf")
    f.write(f"autocorrelation_function :    {time.time() - start:.5f}s\n")
    f.write("\n\n")
