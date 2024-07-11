import numpy as np
from astropy.io import fits
import graphinglib as gl
import time

from summer_2023.gaussian_fitting.fits_analyzer import Map
from src.statistics.stats_library.advanced_stats import autocorrelation_function_cpp, structure_function_cpp

if __name__ == "__main__":
    np.random.seed(1)

    # arr_in = np.random.random((20,20))
    # arr_in[2,1] = np.NAN
    # arr_in[10:14,10:13] = np.NAN

    m = Map(fits.open("summer_2023/gaussian_fitting/maps/computed_data_selective/turbulence.fits"))

    start_1 = time.time()
    cpp = np.array(structure_function_cpp(m.data))
    end_1 = time.time()
    # py = Map(fits.PrimaryHDU(arr_in, None)).get_structure_function_cpp_array()
    start_2 = time.time()
    # py = m.get_structure_function_cpp_array()
    end_2 = time.time()

    cpp_sc = gl.Scatter(cpp[:,0], cpp[:,1], label="Cpp", marker_style="1", face_color="orange")
    # py_sc = gl.Scatter(py[:,0], py[:,1], label="Py", marker_style="2", face_color="lime")

    fig = gl.Figure()
    # fig.add_elements(cpp_sc, py_sc)
    print(f"Cpp time : {end_1-start_1} seconds\nPy time :  {end_2-start_2} seconds")
    fig.show()
