import numpy as np
from astropy.io import fits
import graphinglib as gl

from summer_2023.gaussian_fitting.fits_analyzer import Map
from src.structure_function.advanced_stats import autocorrelation_function, structure_function

if __name__ == "__main__":
    np.random.seed(0)

    arr_in = np.random.random((15,15))
    # print(np.var(arr_in))

    cpp = np.array(structure_function(arr_in))
    py = Map(fits.PrimaryHDU(arr_in, None)).get_structure_function_array()

    cpp_sc = gl.Scatter(cpp[:,0], cpp[:,1], label="Cpp")
    py_sc = gl.Scatter(py[:,0], py[:,1], label="Py")

    fig = gl.Figure()
    fig.add_elements(cpp_sc, py_sc)
    fig.show()
