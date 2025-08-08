import pyregion

from src.tools.deconvolution import *
from src.hdu.cubes.cube import Cube
from src.hdu.maps.map import Map


data_cube = Cube.load("data/orion/data_cubes/nii_1.fits")
calib_cube = Cube.load("data/orion/calibration/calibration_binned.fits")
calib_centroids = Map.load("data/orion/calibration/calibration_centroids.fits")
reg = pyregion.open("data/orion/fp_confidence_regions/nii_1.reg")

n_iterations = 200

deconvolved_data, offsetted_data, offsetted_lsf = deconvolve_cube(
    data_cube,
    calib_cube,
    calib_centroids,
    n_iterations,
)

deconvolution_error = get_deconvolution_error(offsetted_data, offsetted_lsf, deconvolved_data)
Map(deconvolution_error, header=data_cube.header.spectral).get_masked_region(reg)
print(np.nanmean(deconvolution_error.data))

# Map(deconvolution_error).save("data/orion/linewidths/deconvolution/nii_1_deconvolution_error.fits")
# Cube(deconvolved_data).save("data/orion/linewidths/deconvolution/nii_1_deconvolved.fits")
