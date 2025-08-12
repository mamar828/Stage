import numpy as np
import src.graphinglib as gl

from src.hdu.maps.map import Map
from src.hdu.cubes.cube import Cube
from src.tools.deconvolution import deconvolve_cube, get_deconvolution_error


# nii_1_cube = Cube.load("data/orion/linewidths/deconvolution_binned/nii_1_deconvolved.fits")
# sii_1_cube = Cube.load("data/orion/linewidths/deconvolution_binned/sii_1_line_1_deconvolved.fits")

# nii_1_fwhm_channels = Map.load("data/orion/linewidths/maps/nii_1_stddev.fits") * 2 * np.sqrt(2 * np.log(2))
# sii_1_fwhm_channels = Map.load("data/orion/linewidths/maps/sii_1_line_1_stddev.fits") * 2 * np.sqrt(2 * np.log(2))

# nii_1_fwhm_AA = nii_1_fwhm_channels * nii_1_cube.header["XIL"] / 47
# sii_1_fwhm_AA = sii_1_fwhm_channels * sii_1_cube.header["XIL"] / 47

# temperature = 4.73 * 10**4 * (nii_1_fwhm_AA.get_reprojection_on(sii_1_fwhm_AA.header)**2 - sii_1_fwhm_AA**2)

# temperature_plot = temperature.data.plot
# temperature_plot.color_map_range = (0, 1000)
# gl.SmartFigure(elements=[temperature_plot]).show()


# TEST WITH A SINGLE DECONVOLVED SPECTRUM
# ---------------------------------------
nii_global_spec = np.loadtxt("data/orion/linewidths/global_spectrum_tests/nii_1_global_spectrum.dat")
sii_global_spec = np.loadtxt("data/orion/linewidths/global_spectrum_tests/sii_1_global_spectrum.dat")
calib_global_spec = np.loadtxt("data/orion/linewidths/global_spectrum_tests/global_calibration.dat")
random_array = np.random.normal(
    np.mean(sii_global_spec[:10, 1], axis=0),
    np.std(sii_global_spec[:10, 1], axis=0) / 10,
    24,
)
sii_global_spec[:24, 1] = random_array

calib_centroids = 7.5
n_iterations = 100

deconvolved = []
for line_name, global_spec in zip(["nii_1", "sii_1"], [nii_global_spec, sii_global_spec]):
    deconvolved_data, offsetted_data, offsetted_lsf = deconvolve_cube(
        global_spec[:, 1][:, None, None],
        calib_global_spec[:, 1][:, None, None],
        calib_centroids,
        n_iterations,
    )
    deconvolution_error, reconvolved = get_deconvolution_error(offsetted_data, offsetted_lsf, deconvolved_data)
    print(f"Deconvolution error for {line_name}: {np.mean(deconvolution_error)}")

    x_vals = np.arange(48) + 1
    gl.SmartFigure(
        elements=[
            gl.Curve(x_vals, offsetted_data[:,0,0], label="Offsetted Data"),
            gl.Curve(x_vals, offsetted_lsf[:,0,0], label="Offsetted LSF"),
            gl.Curve(x_vals, deconvolved_data[:,0,0], label="Deconvolved Data"),
            gl.Curve(x_vals, reconvolved[:,0,0], label="Reconvolved Data"),
        ],
    ).show()

    deconvolved.append(deconvolved_data)

x_vals = np.arange(48) + 1
nii_deconv_curve = gl.Curve(x_vals, deconvolved[0][:, 0, 0], label="NII Deconvolved Spectrum")
sii_deconv_curve = gl.Curve(x_vals, deconvolved[1][:, 0, 0], label="SII Deconvolved Spectrum")

nii_fit = gl.FitFromGaussian(nii_deconv_curve, guesses=[1, 24, 1])
sii_fit = gl.FitFromGaussian(sii_deconv_curve, guesses=[1, 24, 1])

nii_std = nii_fit.parameters[2]
sii_std = sii_fit.parameters[2]

gl.SmartFigure(
    num_rows=2,
    elements=[[nii_deconv_curve, nii_fit], [sii_deconv_curve, sii_fit]],
    subtitles=[
        f"$\sigma={nii_std:.2f}$, FWHM$={nii_std*2*np.sqrt(2*np.log(2)) * nii_global_spec[0, 0] / 47:.2f}$",
        f"$\sigma={sii_std:.2f}$, FWHM$={sii_std*2*np.sqrt(2*np.log(2)) * sii_global_spec[0, 0] / 47:.2f}$",
    ]
).show()
