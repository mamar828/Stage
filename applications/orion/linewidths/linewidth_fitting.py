import numpy as np
import src.graphinglib as gl
import pyregion
from astropy.modeling.models import Gaussian1D

from src.hdu.cubes.fittable_cube import FittableCube
from src.hdu.tesseract import Tesseract


filenames = [
    "ha_1",
    "ha_2",
    "nii_1",
    "nii_2",
    "oiii_1",
    "oiii_2",
    "sii_1_line_1",
    "sii_2_line_1",
]
# for file in filenames:
#     cube = FittableCube.load(f"data/orion/linewidths/deconvolution_binned/{file}_deconvolved.fits")
#     guesses = cube.range_peak_estimation(ranges=[slice(21,26)])
#     tess = cube.fit(Gaussian1D.evaluate, guesses)
#     tess.save(f"data/orion/linewidths/fits/{file}_deconvolved_fit.fits")

# for file in filenames:
#     tess = Tesseract.load(f"data/orion/linewidths/fits/{file}_deconvolved_fit.fits")
#     stddev_maps = tess.to_grouped_maps().stddev
#     assert len(stddev_maps) == 1, "There should be only one stddev map per tesseract"
#     stddev_maps[0].save(f"data/orion/linewidths/maps/{file}_stddev.fits")





# nii_cube = FittableCube.load("data/orion/linewidths/deconvolution_binned/nii_1_deconvolved.fits")
# nii_cube = nii_cube.get_masked_region(pyregion.open("data/orion/fp_confidence_regions/nii_1.reg")).bin((1, 170, 170), True)
# nii_spec = nii_cube[:,0,0].plot
# nii_fit = gl.FitFromGaussian(nii_spec, guesses=[1, 24, 1])
# nii_std = nii_fit.parameters[2]

# sii_cube = FittableCube.load("data/orion/linewidths/deconvolution_binned/sii_1_line_1_deconvolved.fits")
# sii_cube = sii_cube.get_masked_region(pyregion.open("data/orion/fp_confidence_regions/sii_1.reg")).bin((1, 170, 170), True)
# sii_spec = sii_cube[:,0,0].plot
# sii_fit = gl.FitFromGaussian(sii_spec, guesses=[1, 24, 1])
# sii_std = sii_fit.parameters[2]

# gl.SmartFigure(
#     num_rows=2,
#     elements=[[nii_spec, nii_fit], [sii_spec, sii_fit]],
#     subtitles=[
#         f"$\sigma={nii_std}$, FWHM$={nii_std*2*np.sqrt(2*np.log(2)) * nii_cube.header["XIL"] / 47}$",
#         f"$\sigma={sii_std}$, FWHM$={sii_std*2*np.sqrt(2*np.log(2)) * sii_cube.header["XIL"] / 47}$",
#     ]
# ).show()
