from astropy.io import fits

from fits_analyzer import Data_cube, Map

import matplotlib.pyplot as plt
import numpy as np

"""
In this file are examples of code that have been used to create the fits files. Every operation has been grouped into a function.
"""

# def get_smoothed_instr_f():
#     calibration_cube = 



def get_region_widening_maps():
    global_header = fits.open("gaussian_fitting/maps/computed_data/fwhm_NII.fits")[0]
    

get_region_widening_maps()
