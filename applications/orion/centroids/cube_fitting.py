import src.graphinglib as gl
import numpy as np
from astropy.modeling import models
from collections import namedtuple

from src.hdu.cubes.fittable_cube import FittableCube
from src.hdu.maps.map import Map
from src.hdu.tesseract import Tesseract
from src.coordinates.ds9_coords import DS9Coords
from src.tools.messaging import printt


Info = namedtuple("Info", ["filename", "ranges", "name", "save_filename"])

files = [
    Info("nii_1_binned_3x3.fits", [slice(9, 14), slice(29, 37)], "NII FIELD 1", "nii_1.fits"),
    Info("nii_2_binned_4x4.fits", [slice(9, 14), slice(29, 37)], "NII FIELD 2", "nii_2.fits"),
    Info("oiii_1_binned_3x3.fits", [slice(8, 12)], "OIII FIELD 1", "oiii_1.fits"),
    Info("oiii_2_binned_4x4.fits", [slice(24, 34)], "OIII FIELD 2", "oiii_2.fits"),
    Info("sii_1_binned_3x3.fits", [slice(16, 19), slice(29, 32)], "SII FIELD 1", "sii_1.fits"),
    Info("sii_2_binned_4x4.fits", [slice(16, 22), slice(29, 33)], "SII FIELD 2", "sii_2.fits"),
    Info("ha_2_binned_4x4.fits", [slice(20, 24)], "HA FIELD 2", "ha_2.fits"),
]

def gaussian_model(x, *args):
    return sum([models.Gaussian1D.evaluate(x, amplitude=args[i], mean=args[i+1], stddev=args[i+2])
                for i in range(0, len(args), 3)])
def voigt_model(x, *args):
    return sum([models.Voigt1D().evaluate(x, amplitude_L=args[i], x_0=args[i+1], fwhm_L=args[i+2], fwhm_G=args[i+3])
                for i in range(0, len(args), 4)])

param_bounds = [0, 1, 0, 0], [np.inf, 48, 10, 10]

# for info in files:
#     printt(f"Processing {info.name}...")
#     cube = FittableCube.load(f"data/orion/data_cubes/binned/{info.filename}")
#     guesses = cube.range_peak_estimation(info.ranges, voigt=True)
#     fits = cube.fit(voigt_model, guesses, number_of_parameters=4, maxfev=10000,
#                     bounds=(guesses.shape[0]//4 * param_bounds[0], guesses.shape[0]//4 * param_bounds[1]))
#     save_filename = f"data/orion/fits/{info.save_filename}"
#     fits.save(save_filename)
#     print(f"Saved fits to {save_filename}.", end="\n\n")
# printt("All easy fits processed successfully!")

# HA FIELD 1 (two components)
# ---------------------------
# info = Info("ha_1_binned_3x3.fits", None, "HA FIELD 1", "ha_1.fits")
# printt(f"Processing {info.name}...")
# cube = FittableCube.load(f"data/orion/data_cubes/binned/{info.filename}")
# guesses = cube.find_peaks_estimation(voigt=True, prominence=3, height=15)
# fits = cube.fit(voigt_model, guesses, number_of_parameters=4, maxfev=100000,
#                 bounds=(guesses.shape[0]//4 * param_bounds[0], guesses.shape[0]//4 * param_bounds[1]))
# save_filename = f"data/orion/fits/{info.save_filename}"
# fits.save(save_filename)
# print(f"Saved fits to {save_filename}.", end="\n\n")
# printt("END")
